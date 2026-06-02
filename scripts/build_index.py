"""
构建向量索引：
1. 商品目录 → BGE 向量 → FAISS 混合搜索索引
2. 爆款案例库 → BGE + CLIP 向量 → 多模态检索索引
3. 知识库文档 → BGE 向量 → RAG 索引

运行：python scripts/build_index.py [--mode demo|full]
  demo: 仅文本向量（CPU可跑，约3-5min）
  full: 含CLIP图片向量（建议GPU）
"""
import sys
import json
import numpy as np
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, KNOWLEDGE_DIR
from project2_infra.embedder import Embedder
from project2_infra.hybrid_search import HybridSearchEngine
from project2_infra.multimodal_retrieval import MultimodalCaseRetriever
from project2_infra.reranker import BGEReranker
from project1_app.rag_knowledge import RAGKnowledgeBase


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_product_index(embedder: Embedder):
    products_path = PROCESSED_DIR / "products.json"
    if not products_path.exists():
        logger.error("products.json not found. Run prepare_data.py first.")
        return

    products = load_json(products_path)
    logger.info(f"Encoding {len(products)} products with BGE-M3...")

    texts = [p["text"] for p in products]
    vecs = embedder.encode_text(texts, batch_size=64)

    engine = HybridSearchEngine(index_name="products")
    engine.build(products, vecs)
    logger.info("✅ Product hybrid search index built")


def build_case_index(embedder: Embedder, with_clip: bool = False):
    cases_path = PROCESSED_DIR / "cases.json"
    if not cases_path.exists():
        logger.error("cases.json not found. Run prepare_data.py first.")
        return

    cases = load_json(cases_path)
    logger.info(f"Encoding {len(cases)} cases with BGE-M3...")

    texts = [c["text"] for c in cases]
    text_vecs = embedder.encode_text(texts, batch_size=64)

    image_vecs = None
    if with_clip:
        image_paths = [c["cover_path"] for c in cases if c.get("cover_path")]
        if image_paths:
            logger.info(f"Encoding {len(image_paths)} case cover images with CLIP...")
            image_vecs = embedder.encode_images(image_paths)

    retriever = MultimodalCaseRetriever()
    retriever.build(cases, text_vecs, image_vecs)
    logger.info("✅ Case multimodal retrieval index built")


def build_rag_index(embedder: Embedder, reranker: BGEReranker):
    doc_paths = list(KNOWLEDGE_DIR.glob("*.json"))
    if not doc_paths:
        logger.error("No knowledge docs found. Run prepare_data.py first.")
        return

    documents = []
    for p in doc_paths:
        with open(p, encoding="utf-8") as f:
            documents.append(json.load(f))

    logger.info(f"Building RAG index for {len(documents)} documents...")
    rag = RAGKnowledgeBase(embedder=embedder, reranker=reranker)
    rag.build_from_documents(documents)
    logger.info("✅ RAG knowledge base index built")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "full"], default="demo")
    args = parser.parse_args()

    with_clip = (args.mode == "full")

    logger.info("Loading Embedder (BGE-M3)...")
    embedder = Embedder(use_bge=True, use_clip=with_clip)

    logger.info("Loading Reranker (BGE Reranker)...")
    reranker = BGEReranker()
    reranker._load()

    build_product_index(embedder)
    build_case_index(embedder, with_clip=with_clip)
    build_rag_index(embedder, reranker)

    logger.info("✅ All indices built! Run: streamlit run app/main.py")


if __name__ == "__main__":
    main()
