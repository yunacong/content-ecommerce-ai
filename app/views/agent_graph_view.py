"""LangGraph AI 经营助手页面（多节点状态机版本）"""
from __future__ import annotations

import streamlit as st


def render(components):
    st.title("🔗 AI 经营助手 · LangGraph 版")
    st.caption(
        "基于 LangGraph 状态图编排：意图识别 → 数据诊断 → 指标归因 → 选品推荐 → 评论洞察 → 行动计划 → 报告生成"
    )

    # 延迟 import，避免 LangGraph 未安装时影响整个 app 启动
    try:
        from project1_app.agent_graph.graph import build_graph, run_graph, stream_graph, format_trajectory
        _lg_available = True
    except ImportError:
        _lg_available = False
        st.error("LangGraph 未安装，请执行：`pip install langgraph langchain-core`")
        return

    # 初始化图实例（缓存到 session，避免重复构建）
    if "lg_graph" not in st.session_state:
        ts = components.get("text_to_sql")
        rag = components.get("rag_kb")
        from config import PROCESSED_DIR
        db_path = str(PROCESSED_DIR / "ecommerce.db")
        with st.spinner("初始化 LangGraph 工作流…"):
            st.session_state.lg_graph = build_graph(
                text_to_sql=ts,
                rag_kb=rag,
                db_path=db_path,
            )

    graph = st.session_state.lg_graph

    # 工作流示意图
    with st.expander("📐 工作流节点示意", expanded=False):
        st.code(
            "START\n"
            "  └─ intent_router\n"
            "       ├─ [knowledge] → rag_retrieval → report_generator → END\n"
            "       └─ [others]   → data_diagnosis → metric_attribution\n"
            "                          └─ [有数据] → product_recommendation\n"
            "                                            └─ review_insight\n"
            "                                                 └─ action_planner\n"
            "                                                      └─ report_generator → END",
            language="text",
        )

    # 快速提问
    examples = [
        "最近 7 天 GMV 为什么下滑？帮我推荐接下来重点推的商品",
        "护肤品类转化率低，推几个高潜力 SKU",
        "SKU_023 的用户评价怎么样，内容应该打什么卖点",
        "抖音直播可以引导用户加微信吗？",
        "新品冷启动有哪些运营方法？",
    ]
    st.markdown("**快速提问：**")
    cols = st.columns(len(examples))
    clicked = None
    for i, (col, ex) in enumerate(zip(cols, examples)):
        if col.button(ex, key=f"lg_ex_{i}", use_container_width=True):
            clicked = ex

    st.markdown("---")

    # 历史记录
    if "lg_history" not in st.session_state:
        st.session_state.lg_history = []

    for msg in st.session_state.lg_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("trajectory_text"):
                with st.expander("🔍 执行轨迹", expanded=False):
                    st.code(msg["trajectory_text"], language="text")
            if msg.get("action_tasks"):
                with st.expander("📋 行动计划任务", expanded=False):
                    _render_tasks(msg["action_tasks"])

    # 输入框
    user_input = st.chat_input("输入运营问题…") or clicked
    if not user_input:
        return

    # 渲染用户消息
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.lg_history.append({"role": "user", "content": user_input})

    # 执行 LangGraph（流式展示节点进度）
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        report_placeholder = st.empty()
        trajectory_placeholder = st.empty()

        node_icons = {
            "intent_router": "🎯 意图识别",
            "data_diagnosis": "📊 数据查询",
            "metric_attribution": "🔍 指标归因",
            "product_recommendation": "🛍️ 选品推荐",
            "review_insight": "💬 评论洞察",
            "rag_retrieval": "📚 知识库检索",
            "action_planner": "📋 行动计划",
            "report_generator": "📝 报告生成",
        }

        final_state = None
        try:
            # stream_graph 流式获取节点进度；invoke 获取完整最终状态
            # 两者分开：stream 只用于 UI 进度展示，invoke 才是真实结果
            completed_nodes = []
            for chunk in stream_graph(graph, user_input):
                for node_name in chunk:
                    label = node_icons.get(node_name, node_name)
                    completed_nodes.append(f"✅ {label}")
                    status_placeholder.markdown(
                        "**执行进度：**\n" + " → ".join(completed_nodes[-4:])
                    )
            # stream 完成后用最后一个 chunk 的累积状态（LangGraph stream 最后会输出完整 state）
            # 如果 chunk 格式不含完整 state，直接 invoke 获取
            final_state = run_graph(graph, user_input)

        except Exception as e:
            st.error(f"LangGraph 执行出错: {e}")
            return

        status_placeholder.empty()

        if final_state is None:
            st.warning("未获取到执行结果，请重试。")
            return

        # 最终报告
        report = final_state.get("final_report", "")
        if not report:
            # 兜底拼接
            parts = [
                final_state.get("diagnosis_summary", ""),
                final_state.get("recommendation_text", ""),
                final_state.get("action_plan_text", ""),
            ]
            report = "\n\n".join(p for p in parts if p)

        report_placeholder.markdown(report)

        # 执行轨迹（折叠）
        traj_text = format_trajectory(final_state)
        with trajectory_placeholder.expander("🔍 执行轨迹", expanded=False):
            st.code(traj_text, language="text")

        # 行动计划（表格）
        tasks = final_state.get("action_tasks", [])
        if tasks:
            with st.expander("📋 行动计划任务看板", expanded=True):
                _render_tasks(tasks)

    # 保存到历史
    st.session_state.lg_history.append({
        "role": "assistant",
        "content": report,
        "trajectory_text": traj_text,
        "action_tasks": tasks,
    })


def _render_tasks(tasks: list[dict]):
    import pandas as pd
    if not tasks:
        st.info("无行动任务")
        return

    display_cols = ["task_id", "task_name", "priority", "owner", "deadline",
                    "expected_metric", "review_metric", "status"]
    rows = [{c: t.get(c, "") for c in display_cols} for t in tasks]
    df = pd.DataFrame(rows)

    # 优先级着色
    def color_priority(val):
        colors = {"高": "background-color: #ffcccc", "中": "background-color: #fff3cd", "低": ""}
        return colors.get(val, "")

    st.dataframe(
        df.style.applymap(color_priority, subset=["priority"]),
        use_container_width=True,
        hide_index=True,
    )
