"""AI 选品推荐页面：混合检索 + Reranker + LightGBM + SHAP"""
import streamlit as st
import plotly.express as px
import pandas as pd


def render(components):
    st.title("🛍️ AI 选品推荐")
    st.caption("混合检索（BGE+BM25+RRF）→ BGE Reranker 精排 → LightGBM 个性化排序 → SHAP 可解释性")

    selector = components["product_selector"]
    ranker = components["ranker"]

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        query = st.text_input("选品需求描述", placeholder="例如：夏季防晒护肤适合做短视频投放")
    with col2:
        category = st.selectbox("类目（可选）", ["不限", "美妆护肤", "个护清洁", "彩妆", "香水", "男士护肤"])
    with col3:
        top_k = st.slider("推荐数量", 3, 10, 5)

    if st.button("🔍 智能选品", type="primary", disabled=not query):
        with st.spinner("全链路检索排序中..."):
            result = selector.select(
                query=query,
                category="" if category == "不限" else category,
                top_k=top_k,
            )

        candidates = result["candidates"]
        if not candidates:
            st.warning("暂无匹配商品，尝试修改查询描述")
            return

        st.success(f"找到 {len(candidates)} 个推荐商品")

        # 排序分对比图（归一化到 0-1，lambdarank 原始分可能为负）
        import numpy as np
        lgbm_raw = [c.get("lgbm_score", 0) for c in candidates]
        rerank_raw = [c.get("rerank_score", 0) for c in candidates]
        def norm01(arr):
            mn, mx = min(arr), max(arr)
            if mx == mn:
                return [0.5] * len(arr)
            return [(v - mn) / (mx - mn) for v in arr]
        lgbm_norm = norm01(lgbm_raw)
        rerank_norm = norm01(rerank_raw)

        scores_df = pd.DataFrame([{
            "商品": f"{i+1}. {c.get('title', '')[:12]}",
            "LightGBM排序分（归一化）": lgbm_norm[i],
            "Reranker相关性分（归一化）": rerank_norm[i],
        } for i, c in enumerate(candidates)])

        fig = px.bar(scores_df, x="商品", y=["LightGBM排序分（归一化）", "Reranker相关性分（归一化）"],
                     barmode="group", title="各商品排序得分对比（归一化至0-1）",
                     labels={"value": "得分", "variable": "评分维度"})
        fig.update_layout(height=300, yaxis_range=[0, 1.1])
        st.plotly_chart(fig, use_container_width=True)

        # 商品卡片
        for i, product in enumerate(candidates):
            with st.expander(f"**{i+1}. {product.get('title', 'N/A')}**  |  CTR: {product.get('ctr_hist', product.get('ctr', 0)):.2%}  |  评分: {product.get('rating', 0):.1f}⭐", expanded=(i == 0)):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.metric("类目", product.get("category", "N/A"))
                    st.metric("价格", f"¥{product.get('price', 0):.0f}")
                    st.metric("历史转化率", f"{product.get('cvr_hist', product.get('cvr', 0)):.2%}")

                with c2:
                    if product.get("recommendation_reason"):
                        st.info(f"**推荐理由**\n\n{product['recommendation_reason']}")

                    if product.get("top_features"):
                        st.markdown("**SHAP 特征贡献 Top3**")
                        st.caption("绿色=正向推动排名↑，红色=相对其他商品排名↓（SHAP 为相对值）")
                        feat_map = {
                            "rerank_score": "语义相关性", "ctr_hist": "历史CTR",
                            "cvr_hist": "历史CVR", "rating": "用户评分",
                            "image_sim": "封面视觉", "text_sim": "标题相关度",
                            "rrf_score": "混合检索分", "review_count_log": "评论数量",
                            "price_norm": "价格区间", "category_match": "类目匹配",
                        }
                        for feat, val in product["top_features"][:3]:
                            name = feat_map.get(feat, feat)
                            bar_len = min(int(abs(val) * 15), 15)
                            bar = "▓" * bar_len + "░" * (15 - bar_len)
                            color = "green" if val > 0 else "red"
                            direction = "↑推高排名" if val > 0 else "↓相对低"
                            st.markdown(f":{color}[{'+' if val > 0 else ''}{val:.3f} {direction}] `{name}` {bar}")

        # SHAP 全局重要性（如果有训练好的模型）
        if ranker.is_trained():
            st.markdown("---")
            st.markdown("**模型特征重要性（LightGBM Gain）**")
            imp = ranker.feature_importance()
            if imp:
                imp_df = pd.DataFrame(list(imp.items()), columns=["特征", "重要性"]).sort_values("重要性", ascending=True)
                fig2 = px.bar(imp_df, x="重要性", y="特征", orientation="h",
                              title="LightGBM 特征重要性（Gain）")
                fig2.update_layout(height=350)
                st.plotly_chart(fig2, use_container_width=True)
