#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import random
import re
from math import comb
from pathlib import Path


TASK_TOTALS = {
    "neike": 309,
    "waike": 303,
    "erke": 314,
    "fuchanke": 362,
    "nanke": 341,
    "zhongliuke": 315,
}

CN_TO_TASK = {
    "内科": "neike",
    "外科": "waike",
    "儿科": "erke",
    "妇产科": "fuchanke",
    "男科": "nanke",
    "肿瘤科": "zhongliuke",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse pairwise judge logs and compute TABLE 4 statistics.")
    parser.add_argument("--label", required=True, help="Row label, e.g. 'MeMR vs. Baseline'")
    parser.add_argument("--log", required=True, help="Path to stdout/stderr log or comparison_results_*.txt")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap iterations for 95%% CI")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for bootstrap")
    return parser.parse_args()


def recover_counts_from_percentages(total: int, a_pct: float, b_pct: float, draw_pct: float) -> tuple[int, int, int]:
    approx = [
        int(round(total * a_pct / 100.0)),
        int(round(total * b_pct / 100.0)),
        int(round(total * draw_pct / 100.0)),
    ]
    best = None
    best_err = float("inf")
    for da in range(-2, 3):
        for db in range(-2, 3):
            a = approx[0] + da
            b = approx[1] + db
            d = total - a - b
            if min(a, b, d) < 0:
                continue
            err = (
                abs(a / total * 100.0 - a_pct)
                + abs(b / total * 100.0 - b_pct)
                + abs(d / total * 100.0 - draw_pct)
            )
            if err < best_err:
                best_err = err
                best = (a, b, d)
    if best is None:
        raise ValueError(f"Failed to recover counts from percentages for total={total}")
    return best


def parse_log(path: Path) -> dict[str, dict[str, int]]:
    per_task: dict[str, dict[str, int]] = {}
    current_task: str | None = None

    file_header_re = re.compile(r"文件\s+([a-z]+)\.json\s+对比结果")
    cn_summary_re = re.compile(
        r"文件夹A对比文件夹B《([^：]+)：文件夹A获胜([0-9.]+)%.*?文件夹B获胜([0-9.]+)%.*?平局([0-9.]+)%》"
    )

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            match = file_header_re.search(line)
            if match:
                current_task = match.group(1)
                per_task.setdefault(current_task, {"win": 0, "lose": 0, "tie": 0, "total": 0})
                continue

            if current_task is not None:
                if "文件夹A 获胜:" in line:
                    per_task[current_task]["win"] = int(re.search(r"(\d+)", line).group(1))
                    continue
                if "文件夹B 获胜:" in line:
                    per_task[current_task]["lose"] = int(re.search(r"(\d+)", line).group(1))
                    continue
                if line.startswith("平局:"):
                    per_task[current_task]["tie"] = int(re.search(r"(\d+)", line).group(1))
                    continue
                if "总比较次数:" in line:
                    per_task[current_task]["total"] = int(re.search(r"(\d+)", line).group(1))
                    current_task = None
                    continue

            match = cn_summary_re.search(line)
            if match:
                task = CN_TO_TASK.get(match.group(1))
                if task is None:
                    continue
                a_pct = float(match.group(2))
                b_pct = float(match.group(3))
                draw_pct = float(match.group(4))
                win, lose, tie = recover_counts_from_percentages(TASK_TOTALS[task], a_pct, b_pct, draw_pct)
                per_task.setdefault(task, {"win": win, "lose": lose, "tie": tie, "total": TASK_TOTALS[task]})

    return per_task


def bootstrap_win_rate_ci(win: int, lose: int, tie: int, n_bootstrap: int, seed: int) -> tuple[float, float]:
    total = win + lose + tie
    if total == 0:
        return float("nan"), float("nan")
    samples = [1] * win + [0] * (lose + tie)
    rng = random.Random(seed)
    boot = []
    for _ in range(n_bootstrap):
        boot_sum = 0
        for _ in range(total):
            boot_sum += samples[rng.randrange(total)]
        boot.append(boot_sum / total)
    boot.sort()
    low_idx = max(0, int(math.floor(0.025 * n_bootstrap)) - 1)
    high_idx = min(n_bootstrap - 1, int(math.ceil(0.975 * n_bootstrap)) - 1)
    return boot[low_idx], boot[high_idx]


def exact_two_sided_sign_test(win: int, lose: int) -> float:
    n = win + lose
    if n == 0:
        return 1.0
    k = max(win, lose)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def main() -> int:
    args = parse_args()
    log_path = Path(args.log)
    per_task = parse_log(log_path)
    missing = [task for task in TASK_TOTALS if task not in per_task]
    if missing:
        print(f"WARNING: missing task blocks in log: {', '.join(missing)}")

    win = sum(item["win"] for item in per_task.values())
    lose = sum(item["lose"] for item in per_task.values())
    tie = sum(item["tie"] for item in per_task.values())
    total = win + lose + tie
    if total == 0:
        raise SystemExit("No valid comparisons parsed from log.")

    win_rate = win / total
    ci_low, ci_high = bootstrap_win_rate_ci(win, lose, tie, args.bootstrap, args.seed)
    p_value = exact_two_sided_sign_test(win, lose)

    print(f"Label: {args.label}")
    print(f"Win: {win}")
    print(f"Lose: {lose}")
    print(f"Tie: {tie}")
    print(f"Win-rate: {win_rate * 100:.2f}%")
    print(f"95% CI: [{ci_low * 100:.2f}%, {ci_high * 100:.2f}%]")
    print(f"p-value: {p_value:.6g}")
    print(
        "ROW\t"
        f"{args.label}\t{win}\t{lose}\t{tie}\t"
        f"{win_rate * 100:.2f}%\t[{ci_low * 100:.2f}%, {ci_high * 100:.2f}%]\t{p_value:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
