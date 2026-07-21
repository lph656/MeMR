# build_rag_index.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import json
import argparse
import logging
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 添加项目路径
sys.path.append(".")
sys.path.append("../")
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.insert(0, project_root)

def load_knowledge_base(json_file):
    """从JSON文件加载知识库并转换为LangChain Document对象。"""
    logging.info(f"开始从 '{json_file}' 加载知识库...")
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"错误: 知识库文件未找到 -> {json_file}")
        return []
    except json.JSONDecodeError:
        logging.error(f"错误: 知识库文件格式无效 -> {json_file}")
        return []

    documents = []
    # 假设知识库结构与您提供的RAG示例代码中的一致
    for entity in tqdm(data.get("实体", []), desc="处理知识库实体"):
        disease_name = entity.get("名称", "未知疾病")
        department = entity.get("科室", "未知科室")
        for attribute in entity.get("属性", []):
            attr_name = attribute.get("属性名称", "未知属性")
            content_list = attribute.get("内容", [])
            for content in content_list:
                doc_content = ""
                if isinstance(content, dict) and "问题" in content and "回答" in content:
                    question = content["问题"]
                    answer = content["回答"]
                    doc_content = f"关于 {disease_name} ({attr_name}):\n问题: {question}\n回答: {answer}"
                elif isinstance(content, str):
                    doc_content = f"关于 {disease_name} ({attr_name}): {content}"
                
                if doc_content:
                    documents.append(Document(
                        page_content=doc_content,
                        metadata={
                            "disease": disease_name,
                            "department": department,
                            "attribute": attr_name,
                        }
                    ))
    logging.info(f"知识库加载完成，共生成 {len(documents)} 个文档片段。")
    return documents

def build_and_save_index(kb_path, index_path, embedding_model):
    """构建FAISS索引并保存到本地。"""
    documents = load_knowledge_base(kb_path)
    if not documents:
        logging.error("未能加载任何文档，索引构建终止。")
        return

    logging.info(f"使用嵌入模型 '{embedding_model}' 创建向量...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    logging.info("开始构建FAISS索引，这可能需要一些时间...")
    db = FAISS.from_documents(documents, embeddings)
    
    logging.info(f"索引构建完成，正在保存到 '{index_path}'...")
    db.save_local(index_path)
    logging.info("✅ FAISS索引已成功保存！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为医疗RAG系统构建并保存FAISS索引。")
    parser.add_argument("--kb_path", type=str, required=True, help="知识库来源的JSON文件路径。")
    parser.add_argument("--index_path", type=str, required=True, help="保存FAISS索引的目标目录路径。")
    parser.add_argument("--embedding_model", type=str, required=True, help="用于生成向量的Hugging Face模型名称。")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    if not os.path.exists(args.index_path):
        os.makedirs(args.index_path)
        
    build_and_save_index(args.kb_path, args.index_path, args.embedding_model)
