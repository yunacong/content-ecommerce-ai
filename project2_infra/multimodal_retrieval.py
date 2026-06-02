"""爆款案例多模态检索：CLIP 图片向量 + Sentence-BERT 标题向量 联合检索"""
import numpy as np
import faiss
import pickle
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger
from config import PROCESSED_DIR


class MultimodalCaseRetriever:
    """
    对爆款案例库做图文联合检索。
    每个案例存储两种向量：text_vec（BGE/SBERT）和 image_vec（CLIP）。
    查询时可单独用文本、图片，或图文融合查询。
    """

    def __init__(self):
        self.cases: List[Dict] = []
        self._text_index: faiss.Index = None
        self._image_index: faiss.Index = None
        self._path_prefix = PROCESSED_DIR / "cases"

    def build(self, cases: List[Dict], text_vecs: np.ndarray, image_vecs: Optional[np.ndarray] = None):
        """
        cases: list of dicts with fields: id, title, category, ctr, cvr, cover_path, text
        text_vecs: shape (N, dim)
        image_vecs: shape (N, dim), optional
        """
        self.cases = cases
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        # Text index
        faiss.normalize_L2(text_vecs)
        self._text_index = faiss.IndexFlatIP(text_vecs.shape[1])
        self._text_index.add(text_vecs)
        faiss.write_index(self._text_index, str(self._path_prefix) + "_text.index")

        # Image index
        if image_vecs is not None:
            faiss.normalize_L2(image_vecs)
            self._image_index = faiss.IndexFlatIP(image_vecs.shape[1])
            self._image_index.add(image_vecs)
            faiss.write_index(self._image_index, str(self._path_prefix) + "_image.index")

        with open(str(self._path_prefix) + "_meta.pkl", "wb") as f:
            pickle.dump(self.cases, f)
        logger.info(f"Case retriever built: {len(cases)} cases")

    def load(self):
        self._text_index = faiss.read_index(str(self._path_prefix) + "_text.index")
        image_path = Path(str(self._path_prefix) + "_image.index")
        if image_path.exists():
            self._image_index = faiss.read_index(str(image_path))
        with open(str(self._path_prefix) + "_meta.pkl", "rb") as f:
            self.cases = pickle.load(f)
        logger.info(f"Case retriever loaded: {len(self.cases)} cases")

    def search_by_text(self, query_vec: np.ndarray, top_k: int = 5, min_ctr: float = 0.0) -> List[Dict]:
        query_vec = query_vec.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_vec)
        scores, ids = self._text_index.search(query_vec, top_k * 3)
        return self._build_results(ids[0], scores[0], top_k, min_ctr, score_key="text_score")

    def search_by_image(self, query_vec: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self._image_index is None:
            raise RuntimeError("Image index not built")
        query_vec = query_vec.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_vec)
        scores, ids = self._image_index.search(query_vec, top_k * 2)
        return self._build_results(ids[0], scores[0], top_k, score_key="image_score")

    def search_multimodal(
        self,
        text_vec: np.ndarray,
        image_vec: Optional[np.ndarray] = None,
        top_k: int = 5,
        text_weight: float = 0.5,
        image_weight: float = 0.5,
        min_ctr: float = 0.0,
    ) -> List[Dict]:
        """图文联合检索，RRF 融合"""
        text_vec = text_vec.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(text_vec)
        t_scores, t_ids = self._text_index.search(text_vec, len(self.cases))

        fused = {}
        k = 60
        for rank, (doc_id, score) in enumerate(zip(t_ids[0], t_scores[0])):
            if doc_id < 0:
                continue
            fused[int(doc_id)] = fused.get(int(doc_id), 0) + text_weight / (k + rank + 1)

        if image_vec is not None and self._image_index is not None:
            image_vec = image_vec.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(image_vec)
            i_scores, i_ids = self._image_index.search(image_vec, len(self.cases))
            for rank, (doc_id, score) in enumerate(zip(i_ids[0], i_scores[0])):
                if doc_id < 0:
                    continue
                fused[int(doc_id)] = fused.get(int(doc_id), 0) + image_weight / (k + rank + 1)

        sorted_ids = sorted(fused, key=fused.get, reverse=True)
        results = []
        for doc_id in sorted_ids:
            if doc_id >= len(self.cases):
                continue
            case = dict(self.cases[doc_id])
            if case.get("ctr", 0) < min_ctr:
                continue
            case["fusion_score"] = fused[doc_id]
            case["_id"] = doc_id
            results.append(case)
            if len(results) >= top_k:
                break
        return results

    def _build_results(self, ids, scores, top_k, min_ctr=0.0, score_key="score") -> List[Dict]:
        results = []
        for doc_id, score in zip(ids, scores):
            if doc_id < 0 or doc_id >= len(self.cases):
                continue
            case = dict(self.cases[int(doc_id)])
            if case.get("ctr", 0) < min_ctr:
                continue
            case[score_key] = float(score)
            case["_id"] = int(doc_id)
            results.append(case)
            if len(results) >= top_k:
                break
        return results

    def is_built(self) -> bool:
        return Path(str(self._path_prefix) + "_meta.pkl").exists()
