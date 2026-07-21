import argparse
import json
import os
import sys

from tqdm import tqdm

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.insert(0, project_root)

from utils.inference_utils import generate_memr_response, load_memr_model


def process_dataset(args):
    model, tokenizer, query_encoder, task_list = load_memr_model(
        base_model_path=args.base_model_path,
        checkpoint_dir=args.checkpoint_dir,
        meta_embeddings_path=args.meta_embeddings_path,
        inference_log_dir=args.inference_log_dir,
    )

    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    questions = data.get("questions", [])
    for item in tqdm(questions, desc=f"processing {os.path.basename(args.input_path)}"):
        question_text = item.get("question")
        if not question_text:
            continue

        results.append(
            {
                "id": item.get("id"),
                "question": question_text,
                "answer": generate_memr_response(model, tokenizer, query_encoder, task_list, question_text),
            }
        )

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)

    print(f"Saved generated answers to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate medical answers with a trained MeMR checkpoint.")
    parser.add_argument("--input_path", required=True, help="Input JSON file with a questions list.")
    parser.add_argument("--output_path", required=True, help="Output JSON path.")
    parser.add_argument("--checkpoint_dir", required=True, help="Checkpoint directory with checkpoint_info.json and state_dict.pt.")
    parser.add_argument("--base_model_path", default="chinese-alpaca-plus-7b-hf", help="Base model path.")
    parser.add_argument("--meta_embeddings_path", default="metadata_embeddings/keshi_meta_embeddings.pt", help="Metadata embedding file.")
    parser.add_argument("--inference_log_dir", default="/tmp/memr_inference_logs", help="TensorBoard log dir used by TaskKeyEncoder.")
    process_dataset(parser.parse_args())
