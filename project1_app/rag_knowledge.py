"""RAG 运营知识库：BGE-M3 向量化 + FAISS + BGE Reranker + 引用溯源"""
import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI
from loguru import logger
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, PROCESSED_DIR


class RAGKnowledgeBase:
    def __init__(self, embedder=None, reranker=None):
        self._embedder = embedder
        self._reranker = reranker
        self._client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self._index: faiss.Index = None
        self._chunks: List[Dict] = []
        self._index_path = PROCESSED_DIR / "rag_index.index"
        self._chunks_path = PROCESSED_DIR / "rag_chunks.pkl"

    def build_from_documents(self, documents: List[Dict]):
        """
        documents: list of {title, content, source, doc_type}
        自动切块 → 向量化 → 建 FAISS 索引
        """
        chunks = []
        for doc in documents:
            text_chunks = self._split_chunks(doc["content"], chunk_size=300, overlap=50)
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    "chunk_id": f"{doc.get('id', doc['title'])}_{i}",
                    "text": chunk,
                    "title": doc["title"],
                    "source": doc.get("source", doc["title"]),
                    "doc_type": doc.get("doc_type", "general"),
                })
        self._chunks = chunks
        logger.info(f"Building RAG index: {len(chunks)} chunks")

        texts = [c["text"] for c in chunks]
        vecs = self._embedder.encode_text(texts, batch_size=32)
        faiss.normalize_L2(vecs)

        self._index = faiss.IndexFlatIP(vecs.shape[1])
        self._index.add(vecs)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))
        with open(self._chunks_path, "wb") as f:
            pickle.dump(self._chunks, f)
        logger.info("RAG index built and saved")

    def load(self):
        self._index = faiss.read_index(str(self._index_path))
        with open(self._chunks_path, "rb") as f:
            self._chunks = pickle.load(f)
        logger.info(f"RAG index loaded: {len(self._chunks)} chunks")

    def retrieve(self, query: str, top_k_recall: int = 20, top_k_final: int = 5) -> List[Dict]:
        query_vec = self._embedder.encode_text([query])
        faiss.normalize_L2(query_vec)
        scores, ids = self._index.search(query_vec, top_k_recall)

        candidates = []
        for doc_id, score in zip(ids[0], scores[0]):
            if doc_id < 0:
                continue
            chunk = dict(self._chunks[doc_id])
            chunk["retrieval_score"] = float(score)
            chunk["_id"] = int(doc_id)
            candidates.append(chunk)

        if self._reranker and len(candidates) > top_k_final:
            candidates = self._reranker.rerank(query, candidates, top_k=top_k_final)
        else:
            candidates = candidates[:top_k_final]

        return candidates

    def answer(self, question: str, top_k: int = 5) -> Dict:
        """RAG 增强回答：检索 → 生成 → 附引用来源"""
        chunks = self.retrieve(question, top_k_final=top_k)

        if not chunks:
            return {"answer": "知识库中暂无相关信息。", "sources": [], "chunks": []}

        context = "\n\n".join(
            f"[{i+1}] 来源：{c['source']}\n{c['text']}"
            for i, c in enumerate(chunks)
        )

        prompt = f"""你是一个内容电商运营专家助手，根据以下知识库内容回答商家问题。

知识库内容：
{context}

商家问题：{question}

要求：
1. 只基于知识库内容回答，不要编造
2. 回答要具体实用，3-5句话
3. 在答案末尾加上引用，格式：[来源: 文档名]
4. 如果知识库内容不足以回答，明确说明"""

        resp = self._client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
        )
        answer = resp.choices[0].message.content.strip()

        sources = list(dict.fromkeys(c["source"] for c in chunks))  # 去重保序
        return {"answer": answer, "sources": sources, "chunks": chunks}

    @staticmethod
    def _split_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += chunk_size - overlap
        return chunks

    def is_built(self) -> bool:
        return self._index_path.exists() and self._chunks_path.exists()
