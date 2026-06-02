"""检索质量评估看板：消融实验结果可视化"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import PROCESSED_DIR


def render(components):
    st.title("🔬 检索评估看板")
    st.caption("消融实验：量化每个模块对 Recall@K、MRR、NDCG@K 的贡献")

    ablation_path = PROCESSED_DIR / "ablation_results.csv"

    lgbm_ablation_path = PROCESSED_DIR / "lgbm_ablation.csv"
    tab1, tab2, tab3 = st.tabs(["检索消融实验", "LightGBM 特征消融", "运行新评估"])

    with tab1:
        if ablation_path.exists():
            df = pd.read_csv(ablation_path, index_col=0)

            st.markdown("#### 消融实验结果")
            st.dataframe(df.style.format("{:.4f}").background_gradient(cmap="RdYlGn", axis=0),
                         use_container_width=True)

            # 折线图：各配置的指标变化
            metrics_to_plot = [c for c in df.columns if any(m in c for m in ["Recall", "MRR", "NDCG"])]
            selected_metric = st.selectbox("选择指标", metrics_to_plot)

            fig = go.Figure()
            configs = df.index.tolist()
            values = df[selected_metric].tolist()
            fig.add_trace(go.Scatter(
                x=configs, y=values, mode="lines+markers+text",
                text=[f"{v:.4f}" for v in values],
                textposition="top center",
                line=dict(color="#0066cc", width=3),
                marker=dict(size=10),
            ))
            # 标注最大提升
            baseline = values[0]
            for i, (cfg, val) in enumerate(zip(configs, values)):
                if i > 0:
                    improvement = (val - baseline) / (baseline + 1e-8) * 100
                    fig.add_annotation(
                        x=cfg, y=val,
                        text=f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%",
                        showarrow=False, yshift=25,
                        font=dict(color="green" if improvement > 0 else "red", size=11),
                    )
            fig.update_layout(
                title=f"{selected_metric} 各配置对比",
                xaxis_title="配置", yaxis_title=selected_metric,
                height=400, xaxis_tickangle=-15,
            )
            st.plotly_chart(fig, use_container_width=True)

            # 提升幅度汇总
            st.markdown("#### 最终配置 vs 基线提升幅度")
            baseline_row = df.iloc[0]
            best_row = df.iloc[-1]
            improvements = []
            for col in df.columns:
                imp = (best_row[col] - baseline_row[col]) / (baseline_row[col] + 1e-8) * 100
                improvements.append({"指标": col, "基线": baseline_row[col], "最终": best_row[col], "提升%": imp})
            imp_df = pd.DataFrame(improvements)
            st.dataframe(
                imp_df.style.format({"基线": "{:.4f}", "最终": "{:.4f}", "提升%": "{:+.1f}%"})
                            .background_gradient(subset=["提升%"], cmap="RdYlGn"),
                use_container_width=True,
            )
        else:
            st.info("尚未运行评估。请先运行：`python scripts/run_evaluation.py`")
            st.code("python scripts/run_evaluation.py", language="bash")

    with tab2:
        st.markdown("#### LightGBM 特征消融实验")
        st.caption("三组特征配置：A 纯结构化 → B +文本语义 → C +多模态图片，量化每组特征贡献")
        if lgbm_ablation_path.exists():
            lgbm_df = pd.read_csv(lgbm_ablation_path, index_col=0)
            st.dataframe(lgbm_df.style.format("{:.4f}").background_gradient(cmap="RdYlGn", axis=0),
                         use_container_width=True)

            selected_lgbm = st.selectbox("选择指标", lgbm_df.columns.tolist(), key="lgbm_metric")
            fig_lgbm = go.Figure()
            configs = lgbm_df.index.tolist()
            values = lgbm_df[selected_lgbm].tolist()
            fig_lgbm.add_trace(go.Bar(
                x=configs, y=values,
                text=[f"{v:.4f}" for v in values],
                textposition="outside",
                marker_color=["#4e79a7", "#f28e2b", "#e15759"],
            ))
            baseline = values[0]
            for i in range(1, len(values)):
                imp = (values[i] - baseline) / (baseline + 1e-8) * 100
                fig_lgbm.add_annotation(
                    x=configs[i], y=values[i],
                    text=f"{imp:+.1f}%", showarrow=False, yshift=30,
                    font=dict(color="green" if imp > 0 else "red", size=12),
                )
            fig_lgbm.update_layout(title=f"LightGBM 特征组消融 - {selected_lgbm}", height=380)
            st.plotly_chart(fig_lgbm, use_container_width=True)
        else:
            st.info("尚未训练 LightGBM 模型。请先运行：")
            st.code("python scripts/train_ranker.py", language="bash")

    with tab3:
        st.markdown("#### 运行评估")
        st.markdown("**检索消融实验**（约 3-5 分钟）")
        st.code("python scripts/run_evaluation.py", language="bash")
        st.markdown("**LightGBM 训练 + 特征消融**（约 1-2 分钟）")
        st.code("python scripts/train_ranker.py", language="bash")
        st.warning("两个脚本均需在项目根目录运行")

        if st.button("在后台运行评估（Demo 用途）", type="secondary"):
            import subprocess
            import os
            project_root = Path(__file__).parent.parent.parent
            with st.spinner("评估运行中，约 3-5 分钟..."):
                result = subprocess.run(
                    [sys.executable, "scripts/run_evaluation.py"],
                    capture_output=True, text=True, cwd=str(project_root)
                )
            if result.returncode == 0:
                st.success("评估完成！刷新页面查看结果。")
                st.text(result.stdout[-2000:])
            else:
                st.error("评估出错")
                st.text(result.stderr[-1000:])
