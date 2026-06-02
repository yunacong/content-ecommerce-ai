"""经营数据分析（Text-to-SQL）"""
import streamlit as st


def render(components):
    st.title("📊 经营数据智能分析")
    st.caption("Text-to-SQL：自然语言提问 → 自动生成 SQL → 执行 → 图表 + 根因洞察")

    t2sql = components["text_to_sql"]

    presets = {
        "本周 GMV 趋势": "最近7天每天的GMV是多少？",
        "转化漏斗分析": "最近7天转化漏斗各环节数据（曝光→点击→加购→下单）",
        "CTR 最低 SKU": "哪5个SKU的点击率最低？",
        "GMV Top10 商品": "GMV最高的10个商品是哪些？",
        "类目 GMV 对比": "各类目的总GMV对比",
    }

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**预设查询**")
        preset_cols = st.columns(len(presets))
        selected_preset = None
        for i, (name, q) in enumerate(presets.items()):
            if preset_cols[i].button(name, use_container_width=True):
                selected_preset = q

    question = st.text_input("输入问题", value=selected_preset or "", placeholder="例如：哪个商品本周GMV下滑最严重？")

    if st.button("🔍 分析", type="primary", disabled=not question):
        with st.spinner("生成 SQL 并执行..."):
            result = t2sql.query(question)

        if not result["success"]:
            st.error(f"查询失败：{result.get('error', '未知错误')}")
        else:
            col_sql, col_insight = st.columns([1, 1])
            with col_sql:
                st.markdown("**生成的 SQL**")
                st.code(result["sql"], language="sql")

            with col_insight:
                st.markdown("**AI 洞察**")
                st.info(result["insight"])

            if result["chart"]:
                st.plotly_chart(result["chart"], use_container_width=True)

            st.markdown("**原始数据**")
            st.dataframe(result["data"], use_container_width=True)

    st.markdown("---")
    st.markdown("**技术说明**：LLM (DeepSeek) + Few-Shot Prompting → SQLite → Plotly 可视化 → LLM 洞察生成")
