"""
商品评论洞察工具，供 Agent 调用。

调用方式：
    result = get_review_insight({"sku_id": "SKU_023"})

内部优先调用项目二的 review_analyzer 模块，
若不可用则调用 REMOTE_URL/merchant/product_insight 接口（若已启动）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from loguru import logger


REMOTE_URL = os.getenv("RECOMMENDER_API_URL", "http://localhost:8000")

# 延迟读取，优先使用 config.py（已 load_dotenv），os.getenv 作为兜底
def _get_llm_config() -> tuple[str, str, str]:
    try:
        from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
        return DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
    except ImportError:
        return (
            os.getenv("DEEPSEEK_API_KEY", ""),
            os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        )

# 尝试直接 import 项目二模块（单体部署场景）
_RECSYS_ROOT = Path(__file__).parent.parent.parent.parent / "multimodal-recsys"
if _RECSYS_ROOT.exists():
    sys.path.insert(0, str(_RECSYS_ROOT))

try:
    from product_intelligence.review_analyzer import (
        analyze_reviews, format_insight_text, ReviewInsight
    )
    from product_intelligence.mock_reviews import get_reviews_for_sku
    _LOCAL_AVAILABLE = True
except Exception:
    _LOCAL_AVAILABLE = False


def get_review_insight(params: dict) -> str:
    """
    获取商品评论洞察。

    params:
        sku_id: 商品 ID（必须）
        n_reviews: 分析评论数，默认 10
        use_embedding: 是否启用 SBERT 聚类，默认 False（节省时间）
        use_llm: 是否启用 LLM 总结，默认 True
    """
    sku_id = params.get("sku_id", "")
    if not sku_id:
        return "请提供 sku_id 参数。"

    n = int(params.get("n_reviews", 10))
    use_embedding = bool(params.get("use_embedding", False))
    use_llm = bool(params.get("use_llm", True))

    if _LOCAL_AVAILABLE:
        return _local_insight(sku_id, n, use_embedding, use_llm)

    # 降级：尝试远程接口
    return _remote_insight(sku_id)


def _local_insight(sku_id: str, n: int, use_embedding: bool, use_llm: bool) -> str:
    try:
        reviews = get_reviews_for_sku(sku_id, n=n)
        if not reviews:
            return f"商品 {sku_id} 暂无评论数据。"

        api_key, base_url, model = _get_llm_config()
        insight = analyze_reviews(
            sku_id=sku_id,
            reviews=reviews,
            n_clusters=3,
            use_embedding=use_embedding,
            use_llm=use_llm,
            llm_api_key=api_key,
            llm_base_url=base_url,
            llm_model=model,
        )
        return format_insight_text(insight)
    except Exception as e:
        logger.error(f"评论洞察失败: {e}")
        return f"评论洞察暂时不可用（{e}）。"


def _remote_insight(sku_id: str) -> str:
    try:
        import requests
        resp = requests.post(
            f"{REMOTE_URL}/merchant/product_insight",
            json={"sku_id": sku_id},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        lines = [
            f"**{sku_id} 评论洞察**",
            f"✅ 卖点：{'、'.join(data.get('positive_aspects', []))}",
            f"⚠️ 痛点：{'、'.join(data.get('negative_aspects', []))}",
            f"📢 内容角度：{'、'.join(data.get('content_angles', []))}",
            f"📌 建议：{data.get('recommendation', '')}",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"评论洞察服务不可用，建议参考商品评分和销量数据进行判断。"
