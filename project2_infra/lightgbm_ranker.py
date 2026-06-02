"""LightGBM Learning-to-Rank 个性化排序 + SHAP 特征贡献分析"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from loguru import logger
from config import LGBM_PARAMS, PROCESSED_DIR


FEATURE_COLS = [
    "rerank_score",     # BGE Reranker 相关性分
    "rrf_score",        # 混合检索 RRF 分
    "text_sim",         # 标题语义相似度
    "image_sim",        # 封面视觉相似度
    "price_norm",       # 归一化价格
    "rating",           # 平均评分
    "review_count_log", # log(评论数)
    "category_match",   # 类目匹配
    "ctr_hist",         # 历史点击率
    "cvr_hist",         # 历史转化率
]


class LGBMRanker:
    def __init__(self):
        self._model: lgb.Booster = None
        self._explainer = None
        self._model_path = PROCESSED_DIR / "lgbm_ranker.pkl"

    def build_features(self, candidates: List[Dict], query_meta: Dict = None) -> pd.DataFrame:
        rows = []
        for doc in candidates:
            row = {
                "rerank_score": doc.get("rerank_score", 0.0),
                "rrf_score": doc.get("rrf_score", 0.0),
                "text_sim": doc.get("text_score", doc.get("fusion_score", 0.0)),
                "image_sim": doc.get("image_score", 0.0),
                "price_norm": min(doc.get("price", 50) / 200.0, 1.0),
                "rating": doc.get("rating", 3.5) / 5.0,
                "review_count_log": np.log1p(doc.get("review_count", 10)),
                "category_match": float(doc.get("category", "") == (query_meta or {}).get("category", "")),
                "ctr_hist": doc.get("ctr", 0.05),
                "cvr_hist": doc.get("cvr", 0.02),
            }
            rows.append(row)
        return pd.DataFrame(rows, columns=FEATURE_COLS)

    def train(self, train_data: List[Dict], eval_data: Optional[List[Dict]] = None):
        """
        train_data: list of dicts with 'features', 'label', 'query_id'
        """
        X = pd.DataFrame([d["features"] for d in train_data], columns=FEATURE_COLS)
        y = np.array([d["label"] for d in train_data])
        groups = pd.Series([d["query_id"] for d in train_data]).value_counts().sort_index().values

        ds_train = lgb.Dataset(X, label=y, group=groups, feature_name=FEATURE_COLS)

        callbacks = [lgb.log_evaluation(period=50)]
        if eval_data:
            X_val = pd.DataFrame([d["features"] for d in eval_data], columns=FEATURE_COLS)
            y_val = np.array([d["label"] for d in eval_data])
            grp_val = pd.Series([d["query_id"] for d in eval_data]).value_counts().sort_index().values
            ds_val = lgb.Dataset(X_val, label=y_val, group=grp_val, reference=ds_train)
            self._model = lgb.train(LGBM_PARAMS, ds_train, valid_sets=[ds_val], callbacks=callbacks)
        else:
            self._model = lgb.train(LGBM_PARAMS, ds_train, callbacks=callbacks)

        # 用原生 txt 格式保存（跨平台/跨进程更稳定，避免 pickle+OpenMP 冲突）
        txt_path = self._model_path.with_suffix(".txt")
        self._model.save_model(str(txt_path))
        logger.info(f"LightGBM ranker saved to {txt_path}")

    def load(self):
        txt_path = self._model_path.with_suffix(".txt")
        if txt_path.exists():
            self._model = lgb.Booster(model_file=str(txt_path))
        else:
            # fallback: old pickle format
            with open(self._model_path, "rb") as f:
                self._model = pickle.load(f)

    def rank(self, candidates: List[Dict], query_meta: Dict = None) -> List[Dict]:
        if not candidates:
            return []
        X = self.build_features(candidates, query_meta)
        scores = self._model.predict(X)
        for doc, score in zip(candidates, scores):
            doc["lgbm_score"] = float(score)
        return sorted(candidates, key=lambda x: x["lgbm_score"], reverse=True)

    def explain(self, candidates: List[Dict], query_meta: Dict = None, top_n: int = 3) -> List[Dict]:
        """Add SHAP feature contribution to each candidate"""
        if not candidates:
            return []
        X = self.build_features(candidates, query_meta)

        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self._model)

        shap_values = self._explainer.shap_values(X)

        for i, doc in enumerate(candidates):
            contrib = {feat: float(shap_values[i][j]) for j, feat in enumerate(FEATURE_COLS)}
            top_features = sorted(contrib.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
            doc["shap_contributions"] = contrib
            doc["top_features"] = top_features
        return candidates

    def feature_importance(self) -> Dict:
        if self._model is None:
            return {}
        imp = self._model.feature_importance(importance_type="gain")
        return dict(zip(FEATURE_COLS, imp.tolist()))

    def is_trained(self) -> bool:
        return self._model_path.with_suffix(".txt").exists() or self._model_path.exists()
