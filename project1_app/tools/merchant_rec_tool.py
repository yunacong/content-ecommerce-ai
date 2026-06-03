"""
商家选品推荐工具（双模式）。

USE_REMOTE_RECOMMENDER=true  → 调用 multimodal-recsys FastAPI /merchant/recommend
USE_REMOTE_RECOMMENDER=false → 直接调用本地 project2_infra 推荐链路（Demo 稳定优先）

Agent 通过 _call_tool("merchant_recommend_products", params) 调用此模块。
"""
from __future__ import annotations

import os
import json
from typing import Any

import requests
from loguru import logger


REMOTE_URL = os.getenv("RECOMMENDER_API_URL", "http://localhost:8000")
USE_REMOTE = os.getenv("USE_REMOTE_RECOMMENDER", "false").lower() == "true"
REQUEST_TIMEOUT = int(os.getenv("RECOMMENDER_TIMEOUT", "10"))




def _local_recommend(params: dict) -> str:
    """
    本地模式：直接调用 project2_infra 的 LightGBM 排序链路。

    优先保证 Demo 可运行性，不依赖外部服务。
    """
    try:
        from project2_infra.lightgbm_ranker import LGBMRanker
        from project2_infra.hybrid_search import HybridSearch
        from project2_infra.embedder import Embedder
        from config import PROCESSED_DIR

        embedder = Embedder()
        query = (
            f"{params.get('business_goal', '')} "
            f"{params.get('problem_type', '')} "
            f"{params.get('target_category', '')}"
        ).strip()
        query_vec = embedder.encode_text([query])[0]

        searcher = HybridSearch()
        ranker = LGBMRanker()
        if not ranker.is_trained():
            return _fallback_recommend(params)

        ranker.load()
        candidates = searcher.search(query, query_vec, filters=None)
        if not candidates:
            return _fallback_recommend(params)

        # 价格过滤
        price_range = params.get("price_range", [0, 9999])
        candidates = [
            c for c in candidates
            if price_range[0] <= c.get("price", 50) <= price_range[1]
        ] or candidates  # 过滤后为空则退化

        query_meta = {"category": params.get("target_category", "")}
        ranked = ranker.rank(candidates, query_meta)
        ranked = ranker.explain(ranked, query_meta)

        top_k = params.get("top_k", 10)
        return _format_local(ranked[:top_k], params)

    except Exception as e:
        logger.error(f"本地推荐失败: {e}")
        return _fallback_recommend(params)


def _fallback_recommend(params: dict) -> str:
    """兜底：当本地模型也不可用时，返回提示信息。"""
    goal = params.get("business_goal", "提升GMV")
    category = params.get("target_category", "")
    hint = f"目标：{goal}"
    if category:
        hint += f"，品类：{category}"
    return (
        f"推荐服务当前不可用（{hint}）。\n"
        "建议：1) 检查模型是否已训练；2) 启动 multimodal-recsys FastAPI 服务。\n"
        "可手动查询经营数据，选择 CTR/CVR 历史表现优异的 SKU 重点推广。"
    )


def _format_response(data: dict) -> str:
    """格式化远程服务返回结果。"""
    skus = data.get("recommended_skus", [])
    if not skus:
        return "未找到匹配的推荐商品。"

    goal = data.get("business_goal", "")
    problem = data.get("problem_type", "")
    lines = [f"针对【{problem}】目标【{goal}】，推荐以下 {len(skus)} 个高潜力 SKU：\n"]

    for i, sku in enumerate(skus, 1):
        lines.append(
            f"{i}. {sku['product_name']} (评分: {sku['score']:.3f})\n"
            f"   推荐理由：{sku['reason']}\n"
            f"   内容方向：{sku['content_angle']}\n"
            f"   风险提示：{sku['risk']}"
        )
    return "\n".join(lines)


def _format_local(ranked: list[dict], params: dict) -> str:
    """格式化本地排序结果。"""
    if not ranked:
        return "未找到匹配商品。"

    goal = params.get("business_goal", "")
    lines = [f"本地推荐（目标：{goal}），Top {len(ranked)} SKU：\n"]
    for i, p in enumerate(ranked, 1):
        name = p.get("title", p.get("sku_name", "未知商品"))[:40]
        score = p.get("lgbm_score", p.get("rrf_score", 0))
        reason = p.get("recommendation_reason", "")
        top_feats = p.get("top_features", [])
        feat_str = "、".join(f"{f[0]}({f[1]:+.2f})" for f in top_feats[:2]) if top_feats else ""

        line = f"{i}. {name} | 综合分：{score:.3f}"
        if feat_str:
            line += f" | 主要驱动：{feat_str}"
        if reason:
            line += f"\n   推荐理由：{reason}"
        lines.append(line)

    return "\n".join(lines)


# 模块级缓存：保存最近一次推荐的结构化 SKU 列表，供行动计划生成器使用
_last_recommended_skus: list[dict] = []


def get_last_recommended_skus() -> list[dict]:
    """返回最近一次推荐的结构化 SKU 列表。"""
    return _last_recommended_skus


def _extract_skus_from_remote(data: dict) -> list[dict]:
    """从远程服务响应中提取结构化 SKU 列表。"""
    return [
        {
            "sku_id": s.get("sku_id", ""),
            "product_name": s.get("product_name", ""),
            "content_angle": s.get("content_angle", ""),
            "risk": s.get("risk", ""),
            "score": s.get("score", 0),
        }
        for s in data.get("recommended_skus", [])
    ]


def _extract_skus_from_local(ranked: list[dict]) -> list[dict]:
    """从本地排序结果中提取结构化 SKU 列表。"""
    return [
        {
            "sku_id": p.get("sku_id", p.get("id", f"SKU_{i}")),
            "product_name": p.get("title", p.get("sku_name", "未知商品"))[:40],
            "content_angle": "",
            "risk": "",
            "score": p.get("lgbm_score", p.get("rrf_score", 0)),
        }
        for i, p in enumerate(ranked)
    ]


def recommend_products(params: dict) -> str:
    """
    统一入口，供 Agent._call_tool 调用。
    调用后自动保存结构化结果到 _last_recommended_skus，
    供 generate_action_plan 工具直接读取，无需 LLM 重新解析文本。

    params 示例：
    {
        "business_goal": "提升GMV",
        "problem_type": "CVR下降",
        "target_category": "护肤/面膜",
        "price_range": [50, 150],
        "top_k": 10
    }
    """
    global _last_recommended_skus
    if USE_REMOTE:
        result = _remote_recommend_raw(params)
        if isinstance(result, dict):
            _last_recommended_skus = _extract_skus_from_remote(result)
            return _format_response(result)
        _last_recommended_skus = []
        return result
    else:
        return _local_recommend(params)


def _remote_recommend_raw(params: dict) -> dict | str:
    """调用远程 API，返回原始 dict（成功）或错误字符串（失败）。"""
    payload = {
        "merchant_id": params.get("merchant_id", "m_demo"),
        "business_goal": params.get("business_goal", "提升GMV"),
        "problem_type": params.get("problem_type", "综合指标下滑"),
        "target_category": params.get("target_category", ""),
        "price_range": params.get("price_range", [0, 9999]),
        "top_k": params.get("top_k", 10),
    }
    try:
        resp = requests.post(
            f"{REMOTE_URL}/merchant/recommend",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        logger.warning("推荐服务未启动，自动切换到本地模式")
        return _local_recommend(params)
    except Exception as e:
        logger.error(f"远程推荐服务调用失败: {e}")
        return f"推荐服务暂时不可用，错误：{e}"
