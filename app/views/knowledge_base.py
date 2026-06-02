"""RAG 运营知识库问答"""
import streamlit as st


def render(components):
    st.title("📚 RAG 运营知识库")
    st.caption("BGE-M3 向量检索 + BGE Reranker 精排 + DeepSeek RAG 增强生成 + 引用来源溯源")

    rag = components["rag_kb"]

    presets = [
        "抖音直播间可以引导用户加微信吗？",
        "爆款封面设计有什么规律？",
        "新品没有历史数据怎么评估投放潜力？",
        "GMV 突然下滑应该怎么排查？",
        "短视频脚本钩子怎么设计？",
    ]

    st.markdown("**快速提问**")
    cols = st.columns(len(presets))
    selected = None
    for i, (col, q) in enumerate(zip(cols, presets)):
        if col.button(q[:10] + "...", key=f"rag_{i}", help=q, use_container_width=True):
            selected = q

    question = st.text_input("输入问题", value=selected or "", placeholder="关于平台规则、运营SOP、选品方法的任何问题")

    if st.button("🔍 查询知识库", type="primary", disabled=not question):
        with st.spinner("检索 + 生成中..."):
            result = rag.answer(question)

        st.markdown("### 回答")
        st.write(result["answer"])

        if result["sources"]:
            st.markdown(f"**引用来源**：{' · '.join(result['sources'])}")

        with st.expander("📄 检索到的知识片段"):
            for i, chunk in enumerate(result.get("chunks", [])):
                st.markdown(f"**[{i+1}] {chunk.get('source', '')}**")
                score_label = f"相关性: {chunk.get('rerank_score', chunk.get('retrieval_score', 0)):.4f}"
                st.caption(score_label)
                st.text(chunk["text"][:300])
                st.markdown("---")

    st.markdown("---")
    st.markdown("**知识库内容**：抖音电商运营规则 · 爆款内容创作SOP · 选品方法论 · 运营常见问题FAQ")
