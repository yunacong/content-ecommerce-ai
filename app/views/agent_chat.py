"""AI 经营助手（ReAct Agent）对话界面"""
import streamlit as st


def render(components):
    st.title("🤖 AI 经营助手")
    st.caption("输入任何运营问题，Agent 自动调用数据分析、内容诊断、选品推荐、知识库等工具")

    agent = components["agent"]

    # 示例问题
    examples = [
        "为什么本周 GMV 下滑？",
        "哪些 SKU 的转化率最低，有什么改进建议？",
        "帮我找 3 个美妆类爆款案例，分析共同特征",
        "我要做夏季防晒产品投放，推荐几个高潜力商品",
        "抖音直播中可以引导用户加微信吗？",
    ]

    st.markdown("**快速提问：**")
    cols = st.columns(len(examples))
    clicked_example = None
    for i, (col, ex) in enumerate(zip(cols, examples)):
        if col.button(ex, key=f"ex_{i}", use_container_width=True):
            clicked_example = ex

    st.markdown("---")

    # 对话历史
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 渲染历史
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("trajectory"):
                for step in msg["trajectory"]:
                    tool = step["tool"]
                    obs = str(step.get("result", ""))
                    if len(obs) < 10:
                        continue
                    icon = {"query_data": "📊", "search_cases": "🔍", "select_products": "🛍️",
                            "query_knowledge": "📚", "diagnose_content": "🎨"}.get(tool, "🔧")
                    label = {"query_data": "数据查询结果", "search_cases": "爆款案例",
                             "select_products": "选品推荐", "query_knowledge": "知识库检索"}.get(tool, tool)
                    with st.expander(f"{icon} {label}", expanded=False):
                        if "洞察:" in obs:
                            st.info(obs.split("洞察:")[-1].strip()[:600])
                        elif "知识库回答:" in obs:
                            st.info(obs.split("知识库回答:")[-1].strip()[:600])
                        else:
                            st.text(obs[:600])
                with st.expander(f"⚙️ 执行轨迹（{msg['steps']} 步）", expanded=False):
                    for step in msg["trajectory"]:
                        st.markdown(f"**Step {step['step']}** → `{step['tool']}`")
                        st.code(str(step["params"]), language="python")
                        st.text(str(step["result"])[:500])

    # 输入
    user_input = st.chat_input("输入问题，例如：本周GMV为什么下滑？") or clicked_example
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Agent 思考中..."):
                result = agent.chat(user_input)

            st.markdown(result["answer"])

            # 工具调用结果卡片（直接展示数据，兜底保障）
            if result.get("trajectory"):
                for step in result["trajectory"]:
                    tool = step["tool"]
                    obs = str(step.get("result", ""))
                    if not obs or len(obs) < 10:
                        continue
                    icon = {"query_data": "📊", "search_cases": "🔍",
                            "select_products": "🛍️", "query_knowledge": "📚",
                            "diagnose_content": "🎨"}.get(tool, "🔧")
                    tool_label = {"query_data": "数据查询结果", "search_cases": "爆款案例",
                                  "select_products": "选品推荐", "query_knowledge": "知识库检索",
                                  "diagnose_content": "内容诊断"}.get(tool, tool)
                    with st.expander(f"{icon} {tool_label}", expanded=True):
                        # 提取洞察部分（最有价值）
                        if "洞察:" in obs:
                            insight = obs.split("洞察:")[-1].strip()
                            st.info(insight[:600])
                        elif "知识库回答:" in obs:
                            answer_part = obs.split("知识库回答:")[-1].strip()
                            st.info(answer_part[:600])
                        else:
                            st.text(obs[:600])

                with st.expander(f"⚙️ 执行轨迹（{result['steps']} 步）", expanded=False):
                    for step in result["trajectory"]:
                        st.markdown(f"**Step {step['step']}** → `{step['tool']}`")
                        st.json(step["params"])
                        st.text(str(step["result"])[:300])

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "trajectory": result.get("trajectory", []),
            "steps": result.get("steps", 0),
        })

    # 底部工具栏
    if st.session_state.chat_history:
        col_a, col_b, col_c = st.columns([1, 1, 4])
        if col_a.button("清空对话", type="secondary"):
            st.session_state.chat_history = []
            agent.reset_context(clear_memory=False)  # 保留长期记忆
            st.rerun()
        if col_b.button("清空记忆", type="secondary"):
            st.session_state.chat_history = []
            agent.reset_context(clear_memory=True)
            st.rerun()

    # 侧边栏展示当前长期记忆
    memory = agent.memory_summary
    if memory:
        with st.sidebar.expander(f"🧠 长期记忆（{len(memory)} 条）", expanded=False):
            for m in memory:
                st.caption(m)
