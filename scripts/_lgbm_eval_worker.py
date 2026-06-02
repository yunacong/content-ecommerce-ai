"""
独立 LightGBM 评估 worker：无 PyTorch 依赖。
由 run_evaluation.py 通过子进程调用，结果写入 LGBM_EVAL_TMP 环境变量指定的文件。
"""
import sys
import os
import json
import pickle

import numpy as np
# ⚠️ macOS ARM: LightGBM 必须在 faiss 之前加载，否则 OpenMP 冲突导致 segfault
import lightgbm as lgb
import faiss
from rank_bm25 import BM25Okapi
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── 直接读取配置，不经过 config.py 的 load_dotenv（避免触发其他导入）──
PROCESSED_DIR = ROOT / "data" / "processed"


def load_index():
    index = faiss.read_index(str(PROCESSED_DIR / "products_faiss.index"))
    with open(PROCESSED_DIR / "products_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    documents = meta["documents"]
    bm25 = BM25Okapi(meta["tokenized"])
    return index, documents, bm25


def hybrid_search(query_text, query_vec, faiss_index, documents, bm25, top_k=20):
    query_vec = query_vec.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query_vec)
    scores_dense, ids_dense = faiss_index.search(query_vec, top_k * 2)
    dense_ranks = {int(ids_dense[0][i]): i + 1 for i in range(len(ids_dense[0])) if ids_dense[0][i] >= 0}

    tokens = query_text.lower().split()
    bm25_scores = bm25.get_scores(tokens)
    bm25_top_ids = np.argsort(bm25_scores)[::-1][:top_k * 2]
    bm25_ranks = {int(idx): rank + 1 for rank, idx in enumerate(bm25_top_ids)}

    all_ids = set(dense_ranks.keys()) | set(bm25_ranks.keys())
    k = 60
    rrf_scores = {}
    for doc_id in all_ids:
        score = 0.0
        if doc_id in dense_ranks:
            score += 0.7 / (k + dense_ranks[doc_id])
        if doc_id in bm25_ranks:
            score += 0.3 / (k + bm25_ranks[doc_id])
        rrf_scores[doc_id] = score

    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
    results = []
    for doc_id in sorted_ids:
        if doc_id < len(documents):
            doc = dict(documents[doc_id])
            doc["rrf_score"] = rrf_scores[doc_id]
            doc["rerank_score"] = rrf_scores[doc_id] * 2.0
            doc["_id"] = doc_id
            results.append(doc)
    return results


def lgbm_rank(candidates, model):
    if not candidates:
        return []
    FEATURE_COLS = [
        "rerank_score", "rrf_score", "text_sim", "image_sim",
        "price_norm", "rating", "review_count_log", "category_match",
        "ctr_hist", "cvr_hist",
    ]
    rows = []
    for doc in candidates:
        ctr = doc.get("ctr", 0.05)
        rrf = doc.get("rrf_score", 0.0)
        # 特征构造与训练时保持一致：用 CTR 代理相关性信号
        rerank_score = float(np.clip(ctr * 4.0 + rrf * 2.0, 0, 1))
        text_sim = float(np.clip(ctr * 3.5 + rrf, 0, 1))
        image_sim = float(np.clip(ctr * 3.0, 0, 1))
        row = [
            rerank_score,
            rrf,
            text_sim,
            image_sim,
            min(doc.get("price", 50) / 200.0, 1.0),
            doc.get("rating", 3.5) / 5.0,
            np.log1p(doc.get("review_count", 10)),
            0.0,
            ctr,
            doc.get("cvr", 0.02),
        ]
        rows.append(row)
    import pandas as pd
    X = pd.DataFrame(rows, columns=FEATURE_COLS)
    scores = model.predict(X)
    for doc, score in zip(candidates, scores):
        doc["lgbm_score"] = float(score)
    return sorted(candidates, key=lambda x: x["lgbm_score"], reverse=True)


def recall_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & set(relevant_ids)) / len(relevant_ids)


def mrr(retrieved_ids, relevant_ids):
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids, relevance_map, k):
    dcg = sum(relevance_map.get(doc_id, 0) / np.log2(rank + 2)
              for rank, doc_id in enumerate(retrieved_ids[:k]))
    ideal = sorted(relevance_map.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(rank + 2) for rank, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    tmp_path = os.environ.get("LGBM_EVAL_TMP")
    eval_data_path = os.environ.get("LGBM_EVAL_DATA")
    if not tmp_path or not eval_data_path:
        print("ERROR: Missing env vars", flush=True)
        sys.exit(1)

    with open(eval_data_path, encoding="utf-8") as f:
        eval_data = json.load(f)

    model_path = PROCESSED_DIR / "lgbm_ranker.txt"
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}", flush=True)
        sys.exit(1)

    print("Loading FAISS index...", flush=True)
    faiss_index, documents, bm25 = load_index()

    print("Loading LightGBM model...", flush=True)
    model = lgb.Booster(model_file=str(model_path))

    print(f"Evaluating on {len(eval_data)} queries...", flush=True)
    r5_list, r10_list, mrr_list, ndcg_list = [], [], [], []

    for item in eval_data:
        query = item["query"]
        query_vec = np.array(item["query_vec"], dtype=np.float32)
        relevant_ids = item["relevant_ids"]
        relevance_map = {int(k): v for k, v in item.get("relevance_map", {}).items()}

        candidates = hybrid_search(query, query_vec, faiss_index, documents, bm25)
        ranked = lgbm_rank(candidates, model)
        retrieved_ids = [r["_id"] for r in ranked]

        r5_list.append(recall_at_k(retrieved_ids, relevant_ids, 5))
        r10_list.append(recall_at_k(retrieved_ids, relevant_ids, 10))
        mrr_list.append(mrr(retrieved_ids, relevant_ids))
        ndcg_list.append(ndcg_at_k(retrieved_ids, relevance_map, 10))

    metrics = {
        "Recall@5": float(np.mean(r5_list)),
        "Recall@10": float(np.mean(r10_list)),
        "MRR": float(np.mean(mrr_list)),
        "NDCG@10": float(np.mean(ndcg_list)),
    }
    print(f"Done: {metrics}", flush=True)

    with open(tmp_path, "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
