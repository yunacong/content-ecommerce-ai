"""BGE Reranker cross-encoder 精排：Top-50 → Top-5"""
from typing import List, Dict
from loguru import logger
from config import BGE_RERANKER_MODEL, RERANK_TOP_K


class BGEReranker:
    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            from FlagEmbedding import FlagReranker
            logger.info(f"Loading Reranker: {BGE_RERANKER_MODEL}")
            self._model = FlagReranker(BGE_RERANKER_MODEL, use_fp16=True)

    def rerank(self, query: str, candidates: List[Dict], top_k: int = RERANK_TOP_K) -> List[Dict]:
        """
        candidates: list of dicts with 'text' field
        Returns top_k re-ranked results with 'rerank_score' added.
        """
        if not candidates:
            return []
        self._load()

        pairs = [[query, doc.get("text", doc.get("title", ""))] for doc in candidates]
        scores = self._model.compute_score(pairs, normalize=True)

        if not isinstance(scores, list):
            scores = scores.tolist()

        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
