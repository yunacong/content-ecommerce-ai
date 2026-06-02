"""选品潜力预测：调用 Project2 混合检索 + LightGBM 排序 + SHAP 可解释性"""
from typing import List, Dict, Optional
import numpy as np
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


class ProductSelector:
    def __init__(self, hybrid_search=None, reranker=None, ranker=None, explainer=None, embedder=None):
        self._search = hybrid_search
        self._reranker = reranker
        self._ranker = ranker
        self._explainer = explainer
        self._embedder = embedder
        self._client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    def select(
        self,
        query: str,
        category: str = "",
        budget: float = None,
        top_k: int = 10,
    ) -> Dict:
        """
        全链路选品：混合检索 → Reranker 精排 → LightGBM 排序 → SHAP 解释 → LLM 推荐理由
        """
        query_vec = self._embedder.encode_text([query])[0] if self._embedder else np.zeros(768)

        filters = {}
        if category:
            filters["category"] = category

        # Step1: 混合召回
        candidates = self._search.search(query, query_vec, filters=filters or None)

        # Step2: Reranker 精排
        if self._reranker and len(candidates) > top_k:
            candidates = self._reranker.rerank(query, candidates, top_k=top_k * 2)

        # Step3: LightGBM 排序
        query_meta = {"category": category}
        if self._ranker and self._ranker.is_trained():
            candidates = self._ranker.rank(candidates, query_meta)
            candidates = self._ranker.explain(candidates, query_meta)

        # Step4: 价格过滤
        if budget:
            candidates = [c for c in candidates if c.get("price", 999) <= budget]

        candidates = candidates[:top_k]

        # Step5: LLM 推荐理由
        if self._explainer:
            candidates = self._explainer.batch_generate(query, candidates)

        return {
            "query": query,
            "candidates": candidates,
            "total_found": len(candidates),
        }

    def explain_single(self, product: Dict, query: str) -> str:
        """对单个商品生成详细选品分析"""
        shap_desc = ""
        if "top_features" in product:
            feature_map = {
                "rerank_score": "语义相关性",
                "ctr_hist": "历史点击率",
                "cvr_hist": "历史转化率",
                "rating": "用户口碑",
                "image_sim": "封面视觉吸引力",
                "text_sim": "标题相关度",
            }
            parts = [f"{feature_map.get(f, f)}: {v:+.3f}" for f, v in product["top_features"][:3]]
            shap_desc = "核心得分因素：" + "、".join(parts)

        prompt = f"""商家正在为以下查询场景选品："{query}"

候选商品：{product.get('title', '')}
类目：{product.get('category', '')} | 价格：{product.get('price', 'N/A')} | 评分：{product.get('rating', 'N/A')}
历史 CTR：{product.get('ctr_hist', product.get('ctr', 0)):.2%} | 历史 CVR：{product.get('cvr_hist', product.get('cvr', 0)):.2%}
{shap_desc}

请给出：
1. 这个商品适合做内容投放的核心理由（2句话）
2. 建议的内容方向（封面风格/标题卖点）
3. 潜在风险提示（如果有）

语言简洁，面向运营人员。"""

        resp = self._client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=250,
        )
        return resp.choices[0].message.content.strip()
