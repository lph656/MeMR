from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common import ensure_dir, load_json, setup_logger, timestamp_now, write_environment_snapshot, write_json


SEARCH_ROOT_NAMES = ["checkpoints", "checkpoint", "outputs", "output", "runs", "logs", "saved_models", "models"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover available checkpoints for safety evaluation methods.")
    parser.add_argument("--method_registry", required=True, help="Template method registry JSON.")
    parser.add_argument("--output_dir", default=None, help="Output directory. Defaults to results/high_risk_safety/<timestamp>.")
    return parser.parse_args()


def is_valid_checkpoint_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "checkpoint_info.json")) and os.path.isfile(os.path.join(path, "state_dict.pt"))


def find_search_roots(project_root: str) -> List[str]:
    roots = []
    for root_name in SEARCH_ROOT_NAMES:
        candidate = os.path.join(project_root, root_name)
        if os.path.isdir(candidate):
            roots.append(candidate)
    special_candidates = [
        os.path.join(project_root, "checkpoints_continual_keshi_llama"),
        os.path.join(project_root, "metadata_robustness_experiments", "outputs"),
    ]
    for candidate in special_candidates:
        if os.path.isdir(candidate) and candidate not in roots:
            roots.append(candidate)
    return roots


def collect_checkpoint_candidates(search_roots: List[str]) -> List[str]:
    candidates = []
    for root in search_roots:
        for dirpath, _, _ in os.walk(root):
            if is_valid_checkpoint_dir(dirpath):
                candidates.append(dirpath)
    return sorted(set(candidates))


def match_method_candidates(method: Dict[str, Any], candidates: List[str]) -> List[str]:
    explicit = method.get("checkpoint")
    if explicit:
        explicit_path = os.path.abspath(explicit)
        return [explicit_path] if is_valid_checkpoint_dir(explicit_path) else []

    aliases = [alias.lower() for alias in method.get("aliases", [])]
    name_lower = method["name"].lower()
    method_candidates = []
    for candidate in candidates:
        rel_candidate = os.path.relpath(candidate, PROJECT_ROOT)
        candidate_lower = rel_candidate.lower()
        path_parts = [part.lower() for part in rel_candidate.split(os.sep)]
        base_name = os.path.basename(candidate_lower)
        parent_name = os.path.basename(os.path.dirname(candidate_lower))
        haystack = " ".join(path_parts + [base_name, parent_name])

        if name_lower == "memr":
            if "compose_peft" in haystack or "memr" in haystack:
                method_candidates.append(candidate)
            continue

        if any(alias in haystack for alias in aliases):
            method_candidates.append(candidate)
    return sorted(set(method_candidates), key=lambda path: os.path.getmtime(path), reverse=True)


def build_missing_entry(method: Dict[str, Any], reason: str, all_candidates: List[str]) -> Dict[str, Any]:
    return {
        "name": method["name"],
        "aliases": method.get("aliases", []),
        "reason": reason,
        "all_candidate_checkpoints_scanned": all_candidates,
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "results", "high_risk_safety", timestamp_now())
    ensure_dir(output_dir)
    logger = setup_logger("high_risk_safety.discover", output_dir)
    write_environment_snapshot(output_dir)

    registry = load_json(args.method_registry)
    methods = registry.get("methods", [])
    search_roots = find_search_roots(PROJECT_ROOT)
    logger.info("Search roots: %s", search_roots)
    all_candidates = collect_checkpoint_candidates(search_roots)
    logger.info("Discovered %d candidate checkpoints.", len(all_candidates))

    resolved_methods = []
    missing = []
    for method in methods:
        if not method.get("enabled", True):
            logger.info("Skipping disabled method: %s", method["name"])
            method_copy = dict(method)
            method_copy["resolved_candidates"] = []
            method_copy["resolved_checkpoint"] = None
            resolved_methods.append(method_copy)
            continue

        method_candidates = match_method_candidates(method, all_candidates)
        logger.info("Method %s candidates: %s", method["name"], method_candidates)
        method_copy = dict(method)
        method_copy["resolved_candidates"] = method_candidates
        method_copy["resolved_checkpoint"] = method_candidates[0] if method_candidates else None
        resolved_methods.append(method_copy)

        if not method_candidates:
            missing.append(build_missing_entry(method, "checkpoint_not_found", all_candidates))

    resolved_payload = {
        "note": "This safety extension is inference-only and uses existing checkpoints without retraining.",
        "search_roots": search_roots,
        "methods": resolved_methods,
    }
    write_json(os.path.join(output_dir, "method_registry.resolved.json"), resolved_payload)
    write_json(os.path.join(output_dir, "missing_checkpoints.json"), missing)
    logger.info("Saved resolved registry and missing checkpoint report to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
