"""LLM 生成可解释推荐理由 + 证据引用"""
from typing import List, Dict, Tuple
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


class RecommendationExplainer:
    def __init__(self):
        self._client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    def generate_reason(self, query: str, candidate: Dict, shap_top: List[Tuple] = None) -> str:
        """为单个推荐结果生成自然语言解释"""
        title = candidate.get("title", "该商品")
        ctr = candidate.get("ctr_hist", candidate.get("ctr", 0))
        rating = candidate.get("rating", 0)
        category = candidate.get("category", "")
        rerank_score = candidate.get("rerank_score", 0)

        feature_desc = ""
        if shap_top:
            feature_map = {
                "rerank_score": "语义相关性",
                "image_sim": "封面视觉相似度",
                "text_sim": "标题语义相似度",
                "ctr_hist": "历史点击率",
                "cvr_hist": "历史转化率",
                "rating": "用户评分",
                "review_count_log": "评论数量",
                "category_match": "类目匹配度",
            }
            top_feats = [f"{feature_map.get(f, f)}({v:+.3f})" for f, v in shap_top[:3]]
            feature_desc = f"主要贡献特征：{', '.join(top_feats)}"

        prompt = f"""你是一个内容电商 AI 经营助手，为商家解释推荐理由。

商家查询：{query}
推荐商品：{title}（类目：{category}，评分：{rating:.1f}，历史CTR：{ctr:.2%}）
相关性分数：{rerank_score:.3f}
{feature_desc}

请用2-3句话，简洁地解释为什么推荐这个商品，要有具体数据支撑，避免空话。直接输出解释，不要前缀。"""

        resp = self._client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150,
        )
        return resp.choices[0].message.content.strip()

    def batch_generate(self, query: str, candidates: List[Dict]) -> List[Dict]:
        for doc in candidates:
            shap_top = doc.get("top_features")
            doc["recommendation_reason"] = self.generate_reason(query, doc, shap_top)
        return candidates
