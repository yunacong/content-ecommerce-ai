"""内容诊断页面：标题评分 + 封面评分 + 脚本分析"""
import streamlit as st
import plotly.graph_objects as go


def render_score_gauge(score: float, label: str):
    color = "#28a745" if score >= 0.7 else "#ffc107" if score >= 0.4 else "#dc3545"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        title={"text": label, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#ffe0e0"},
                {"range": [40, 70], "color": "#fff8e0"},
                {"range": [70, 100], "color": "#e0f8e0"},
            ],
        },
        number={"suffix": "分", "font": {"size": 28}},
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def render(components):
    st.title("🎨 智能内容诊断")
    st.caption("CLIP 封面视觉评分 + BGE 标题语义评分 + LLM 脚本结构分析")

    diagnosis = components["content_diagnosis"]

    tab1, tab2, tab3 = st.tabs(["标题诊断", "封面诊断", "脚本分析"])

    with tab1:
        st.markdown("#### 标题语义评分")
        st.caption("与爆款案例库做语义距离对比，量化标题质量")
        title = st.text_input("输入内容标题", placeholder="例如：玻尿酸精华液补水保湿")
        category = st.selectbox("所属类目", ["美妆护肤", "个护清洁", "彩妆", "香水", "男士护肤"])

        if st.button("诊断标题", type="primary", disabled=not title):
            with st.spinner("正在分析..."):
                result = diagnosis.score_title(title, category)

            col1, col2 = st.columns([1, 2])
            with col1:
                st.plotly_chart(render_score_gauge(result["score"], "标题得分"), use_container_width=True)

            with col2:
                st.metric("语义相似度（Top1）", f"{result.get('top_similarity', 0):.3f}")
                st.metric("对标爆款平均 CTR", f"{result.get('benchmark_ctr', 0):.2%}")
                if result.get("similar_titles"):
                    st.markdown("**相似爆款标题**")
                    for item in result["similar_titles"]:
                        st.markdown(f"- {item['title']} *(CTR: {item['ctr']:.2%})*")

    with tab2:
        st.markdown("#### 封面视觉评分")
        st.caption("full 模式：CLIP embedding → 爆款封面库向量对比 | demo 模式：LLM 多模态视觉分析")
        uploaded = st.file_uploader("上传封面图片", type=["jpg", "jpeg", "png"])

        if uploaded:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
            st.image(img, caption="待诊断封面", width=300)

            if st.button("诊断封面", type="primary"):
                mode_hint = "CLIP 向量化 + 爆款库对比中..." if diagnosis._has_image_index() else "LLM 视觉分析中（demo 模式）..."
                with st.spinner(mode_hint):
                    result = diagnosis.score_cover(img)

                mode = result.get("mode", "clip")

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.plotly_chart(render_score_gauge(result["score"], "封面得分"), use_container_width=True)

                with col2:
                    if mode == "clip":
                        st.metric("视觉相似度（Top1）", f"{result.get('top_similarity', 0):.3f}")
                        st.metric("对标爆款平均 CTR", f"{result.get('benchmark_ctr', 0):.2%}")
                        if result.get("similar_cases"):
                            st.markdown("**视觉相似爆款**")
                            for c in result["similar_cases"][:3]:
                                st.markdown(f"- {c.get('title', 'N/A')} *(CTR: {c.get('ctr', 0):.2%})*")

                    elif mode == "llm_vision":
                        st.caption("📌 当前为 LLM 视觉分析（demo 模式）；部署 full 模式可换 CLIP 向量对标")
                        dims = result.get("dimensions", {})
                        if dims:
                            dim_cols = st.columns(len(dims))
                            for col, (name, val) in zip(dim_cols, dims.items()):
                                col.metric(name, f"{val}/10")
                        if result.get("issues"):
                            for issue in result["issues"]:
                                st.error(f"⚠️ {issue}")
                        if result.get("suggestions"):
                            for sug in result["suggestions"]:
                                st.success(f"✅ {sug}")

                    else:
                        st.warning(result.get("tip", "分析失败，请检查 API Key 是否支持多模态"))

    with tab3:
        st.markdown("#### 脚本结构分析")
        st.caption("LLM 分析钩子、卖点表达、行动号召的结构质量")
        script = st.text_area("粘贴视频脚本", height=200,
                              placeholder="0-3秒：开场白...\n卖点介绍...\n行动号召...")

        if st.button("分析脚本", type="primary", disabled=not script):
            with st.spinner("LLM 分析中..."):
                result = diagnosis.analyze_script(script)

            cols = st.columns(4)
            metrics = [
                ("钩子强度", result.get("hook_score", 0)),
                ("卖点完整性", result.get("selling_point_score", 0)),
                ("行动号召", result.get("cta_score", 0)),
                ("整体结构", result.get("structure_score", 0)),
            ]
            for col, (name, score) in zip(cols, metrics):
                col.plotly_chart(render_score_gauge(score / 10, name), use_container_width=True)

            if result.get("issues"):
                st.markdown("**主要问题**")
                for issue in result["issues"]:
                    st.error(f"⚠️ {issue}")

            if result.get("suggestions"):
                st.markdown("**改进建议**")
                for sug in result["suggestions"]:
                    st.success(f"✅ {sug}")
