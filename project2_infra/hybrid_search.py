"""混合检索：BGE 稠密向量 + BM25 稀疏关键词 + RRF 融合"""
import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from loguru import logger
from config import HYBRID_TOP_K, BM25_WEIGHT, BGE_WEIGHT, PROCESSED_DIR


class HybridSearchEngine:
    def __init__(self, index_name: str = "products"):
        self.index_name = index_name
        self.faiss_index: faiss.Index = None
        self.bm25: BM25Okapi = None
        self.documents: List[Dict] = []
        self._index_path = PROCESSED_DIR / f"{index_name}_faiss.index"
        self._meta_path = PROCESSED_DIR / f"{index_name}_meta.pkl"

    def build(self, documents: List[Dict], embeddings: np.ndarray):
        """
        documents: list of dicts with at least 'id', 'title', 'text' fields
        embeddings: BGE dense vectors, shape (N, dim)
        """
        self.documents = documents
        logger.info(f"Building FAISS index for {len(documents)} documents")
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)  # inner product (vectors must be L2-normalized)
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)

        logger.info("Building BM25 index")
        tokenized = [doc["text"].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(self._index_path))
        with open(self._meta_path, "wb") as f:
            pickle.dump({"documents": self.documents, "tokenized": tokenized}, f)
        logger.info(f"Index saved to {self._index_path}")

    def load(self):
        self.faiss_index = faiss.read_index(str(self._index_path))
        with open(self._meta_path, "rb") as f:
            meta = pickle.load(f)
        self.documents = meta["documents"]
        self.bm25 = BM25Okapi(meta["tokenized"])
        logger.info(f"Loaded index: {len(self.documents)} documents")

    def search(
        self,
        query_text: str,
        query_vec: np.ndarray,
        top_k: int = HYBRID_TOP_K,
        filters: Dict = None,
    ) -> List[Dict]:
        """
        Returns top_k results fused via Reciprocal Rank Fusion.
        query_vec: shape (1, dim) or (dim,), L2-normalized BGE embedding
        """
        query_vec = query_vec.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_vec)

        # Dense retrieval
        scores_dense, ids_dense = self.faiss_index.search(query_vec, top_k * 2)
        dense_ranks = {int(ids_dense[0][i]): i + 1 for i in range(len(ids_dense[0])) if ids_dense[0][i] >= 0}

        # Sparse retrieval
        tokens = query_text.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_top_ids = np.argsort(bm25_scores)[::-1][: top_k * 2]
        bm25_ranks = {int(idx): rank + 1 for rank, idx in enumerate(bm25_top_ids)}

        # RRF fusion
        all_ids = set(dense_ranks.keys()) | set(bm25_ranks.keys())
        rrf_scores = {}
        k = 60  # RRF constant
        for doc_id in all_ids:
            score = 0.0
            if doc_id in dense_ranks:
                score += BGE_WEIGHT / (k + dense_ranks[doc_id])
            if doc_id in bm25_ranks:
                score += BM25_WEIGHT / (k + bm25_ranks[doc_id])
            rrf_scores[doc_id] = score

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        results = []
        for doc_id in sorted_ids:
            if doc_id >= len(self.documents):
                continue
            doc = dict(self.documents[doc_id])
            doc["rrf_score"] = rrf_scores[doc_id]
            doc["dense_rank"] = dense_ranks.get(doc_id)
            doc["bm25_rank"] = bm25_ranks.get(doc_id)
            doc["_id"] = doc_id

            # Apply metadata filters
            if filters:
                if not all(doc.get(k) == v for k, v in filters.items()):
                    continue
            results.append(doc)

        return results[:top_k]

    def is_built(self) -> bool:
        return self._index_path.exists() and self._meta_path.exists()
