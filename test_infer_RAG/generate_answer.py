import argparse
import json
import os
import sys

import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.insert(0, project_root)

from utils.inference_utils import generate_memr_response, load_memr_model


def load_rag_retriever(index_path: str, embedding_model: str, top_k: int):
    if not os.path.isdir(index_path):
        raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": top_k})


def build_rag_prompt(question: str, context: str) -> str:
    escaped_context = context.replace("{", "{{").replace("}", "}}")
    return f"""你是一位专业的AI医生。请严格根据以下背景知识回答患者提问；如果背景知识无法回答，则根据医学知识谨慎回复。

### 背景知识:
{escaped_context}

### 患者提问:
{question}

### AI医生回答:"""


def generate_rag_response(model, tokenizer, query_encoder, task_list, retriever, question: str):
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    return generate_memr_response(
        model=model,
        tokenizer=tokenizer,
        query_encoder=query_encoder,
        task_list=task_list,
        question=question,
        prompt_template=build_rag_prompt(question="{question}", context=context),
    )


def process_dataset(args):
    model, tokenizer, query_encoder, task_list = load_memr_model(
        base_model_path=args.base_model_path,
        checkpoint_dir=args.checkpoint_dir,
        meta_embeddings_path=args.meta_embeddings_path,
        inference_log_dir=args.inference_log_dir,
    )
    retriever = load_rag_retriever(args.index_path, args.embedding_model, args.top_k)

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
                "answer": generate_rag_response(model, tokenizer, query_encoder, task_list, retriever, question_text),
            }
        )

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)

    print(f"Saved RAG generated answers to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RAG-enhanced medical answers with a trained MeMR checkpoint.")
    parser.add_argument("--input_path", required=True, help="Input JSON file with a questions list.")
    parser.add_argument("--output_path", required=True, help="Output JSON path.")
    parser.add_argument("--checkpoint_dir", required=True, help="Checkpoint directory with checkpoint_info.json and state_dict.pt.")
    parser.add_argument("--index_path", required=True, help="Prebuilt FAISS index directory.")
    parser.add_argument("--embedding_model", required=True, help="Embedding model for retrieval.")
    parser.add_argument("--base_model_path", default="chinese-alpaca-plus-7b-hf", help="Base model path.")
    parser.add_argument("--meta_embeddings_path", default="metadata_embeddings/keshi_meta_embeddings.pt", help="Metadata embedding file.")
    parser.add_argument("--inference_log_dir", default="/tmp/memr_inference_logs", help="TensorBoard log dir used by TaskKeyEncoder.")
    parser.add_argument("--top_k", type=int, default=1, help="Number of retrieved documents.")
    process_dataset(parser.parse_args())
