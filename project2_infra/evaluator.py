"""检索质量评估体系：Recall@K、MRR、NDCG@K、AUC + 消融实验"""
import numpy as np
import pandas as pd
from typing import List, Dict, Callable, Optional
from loguru import logger


class SearchEvaluator:
    @staticmethod
    def recall_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
        if not relevant_ids:
            return 0.0
        hits = len(set(retrieved_ids[:k]) & set(relevant_ids))
        return hits / len(relevant_ids)

    @staticmethod
    def mrr(retrieved_ids: List[int], relevant_ids: List[int]) -> float:
        relevant_set = set(relevant_ids)
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: List[int], relevance_map: Dict[int, int], k: int) -> float:
        """relevance_map: {doc_id: relevance_score (0/1/2)}"""
        dcg = sum(
            relevance_map.get(doc_id, 0) / np.log2(rank + 2)
            for rank, doc_id in enumerate(retrieved_ids[:k])
        )
        ideal_rels = sorted(relevance_map.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(rank + 2) for rank, rel in enumerate(ideal_rels))
        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_retrieval(
        self,
        search_fn: Callable,
        eval_set: List[Dict],
        k_values: List[int] = [5, 10],
    ) -> Dict:
        """
        eval_set: list of dicts with 'query', 'query_vec', 'relevant_ids', 'relevance_map'
        search_fn: callable(query_text, query_vec) -> list of dicts with '_id'
        """
        results = {f"Recall@{k}": [] for k in k_values}
        results["MRR"] = []
        results[f"NDCG@{max(k_values)}"] = []

        for sample in eval_set:
            retrieved = search_fn(sample["query"], sample.get("query_vec"))
            retrieved_ids = [r["_id"] for r in retrieved]
            relevant_ids = sample["relevant_ids"]
            relevance_map = sample.get("relevance_map", {rid: 1 for rid in relevant_ids})

            for k in k_values:
                results[f"Recall@{k}"].append(self.recall_at_k(retrieved_ids, relevant_ids, k))
            results["MRR"].append(self.mrr(retrieved_ids, relevant_ids))
            results[f"NDCG@{max(k_values)}"].append(
                self.ndcg_at_k(retrieved_ids, relevance_map, max(k_values))
            )

        return {k: float(np.mean(v)) for k, v in results.items()}

    def ablation_study(
        self,
        eval_set: List[Dict],
        configurations: Dict[str, Callable],
        k_values: List[int] = [5, 10],
    ) -> pd.DataFrame:
        """
        configurations: {'BM25 only': fn, 'BGE only': fn, 'Hybrid+RRF': fn, ...}
        Returns DataFrame with one row per config and metric columns.
        """
        rows = []
        for config_name, search_fn in configurations.items():
            logger.info(f"Evaluating: {config_name}")
            metrics = self.evaluate_retrieval(search_fn, eval_set, k_values)
            row = {"配置": config_name, **metrics}
            rows.append(row)
            logger.info(f"  {metrics}")
        df = pd.DataFrame(rows).set_index("配置")
        return df

    @staticmethod
    def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
