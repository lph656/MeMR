import argparse
import os
import sys

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.insert(0, project_root)

from utils.inference_utils import generate_memr_response, load_memr_model


def main():
    parser = argparse.ArgumentParser(description="Interactive MeMR medical consultation inference.")
    parser.add_argument("--checkpoint_dir", required=True, help="Checkpoint directory with checkpoint_info.json and state_dict.pt.")
    parser.add_argument("--base_model_path", default="chinese-alpaca-plus-7b-hf", help="Base model path.")
    parser.add_argument("--meta_embeddings_path", default="metadata_embeddings/keshi_meta_embeddings.pt", help="Metadata embedding file.")
    parser.add_argument("--inference_log_dir", default="/tmp/memr_inference_logs", help="TensorBoard log dir used by TaskKeyEncoder.")
    args = parser.parse_args()

    model, tokenizer, query_encoder, task_list = load_memr_model(
        base_model_path=args.base_model_path,
        checkpoint_dir=args.checkpoint_dir,
        meta_embeddings_path=args.meta_embeddings_path,
        inference_log_dir=args.inference_log_dir,
    )

    print("MeMR medical consultation model is ready. Enter 'q' to exit.")
    while True:
        question = input("\n请输入您的问题: \n> ").strip()
        if question.lower() == "q":
            break
        answer = generate_memr_response(model, tokenizer, query_encoder, task_list, question)
        print("\nAI医生回答:")
        print(answer)


if __name__ == "__main__":
    main()
