"""HuggingFace Spaces 入口 — 内容电商 AI 增长平台"""
import sys
import os
import subprocess
import traceback
from pathlib import Path

# 项目根目录 = 本文件所在目录
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")

import streamlit as st

st.set_page_config(
    page_title="内容电商 AI 增长平台",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 冷启动检测 ────────────────────────────────────────────────────────
PROCESSED = ROOT / "data" / "processed"

def _is_ready() -> bool:
    return (PROCESSED / "ecommerce.db").exists() and \
           (PROCESSED / "products_faiss.index").exists() and \
           (PROCESSED / "rag_index.index").exists()

if not _is_ready():
    st.markdown("## 🚀 首次启动，正在初始化...")
    st.info("HuggingFace Spaces 冷启动需约 **5-8 分钟** 下载模型 + 构建索引，请耐心等待。")
    progress = st.progress(0, text="准备数据...")

    with st.spinner("Step 1/2：生成模拟数据..."):
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "prepare_data.py"), "--mode", "demo"],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        if r.returncode != 0:
            st.error(f"数据准备失败：{r.stderr[-500:]}")
            st.stop()
    progress.progress(30, text="数据准备完成，正在下载模型并构建向量索引...")

    with st.spinner("Step 2/2：下载模型 + 构建向量索引（约 4-6 分钟）..."):
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_index.py"), "--mode", "demo"],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        if r.returncode != 0:
            st.error(f"索引构建失败：{r.stderr[-500:]}")
            st.stop()
    progress.progress(100, text="✅ 初始化完成！")

    st.success("初始化完成！页面将自动刷新...")
    st.balloons()
    import time; time.sleep(2)
    st.rerun()

# ── 加载组件 ──────────────────────────────────────────────────────────
try:
    from app.components.loader import load_all_components
except Exception as e:
    import streamlit as st
    st.error(f"模块加载失败：{e}")
    st.code(traceback.format_exc())
    st.stop()

# ── 全局样式 ──────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.stApp { background: #f0f4f8; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e3a5f 100%) !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }
[data-testid="stSidebar"] hr { border-color: #334155 !important; }
.main-card {
    background: #ffffff; border-radius: 12px; padding: 24px; margin: 12px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 4px 16px rgba(0,0,0,0.04);
    border: 1px solid #e2e8f0;
}
[data-testid="metric-container"] {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 16px !important; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="stMetricValue"] { color: #1e3a5f !important; font-weight: 700; }
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    border: none !important; border-radius: 8px !important; color: white !important;
    font-weight: 600 !important; padding: 0.5rem 1.5rem !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important;
}
[data-testid="stButton"] button[kind="secondary"] {
    border-radius: 8px !important; border: 1px solid #cbd5e1 !important; color: #475569 !important;
}
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
    border-radius: 8px !important; border: 1px solid #cbd5e1 !important; background: #f8fafc !important;
}
[data-testid="stSuccess"] { border-radius: 8px; border-left: 4px solid #10b981; }
[data-testid="stWarning"] { border-radius: 8px; border-left: 4px solid #f59e0b; }
[data-testid="stError"]   { border-radius: 8px; border-left: 4px solid #ef4444; }
[data-testid="stInfo"]    { border-radius: 8px; border-left: 4px solid #2563eb; }
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important; border-radius: 10px !important; background: #ffffff !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #2563eb !important; border-bottom: 2px solid #2563eb !important; font-weight: 600;
}
.score-high {color: #10b981; font-weight: 700;}
.score-mid  {color: #f59e0b; font-weight: 700;}
.score-low  {color: #ef4444; font-weight: 700;}
h1 { color: #0f172a !important; font-weight: 800 !important; letter-spacing: -0.5px; }
h2 { color: #1e3a5f !important; font-weight: 700 !important; }
h3 { color: #334155 !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

components = load_all_components()

st.sidebar.markdown("""
<div style="padding: 8px 0 16px 0;">
  <div style="font-size: 22px; font-weight: 800; color: #f1f5f9; letter-spacing: -0.5px;">
    🚀 内容电商 AI 平台
  </div>
  <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">
    Content E-commerce AI Growth
  </div>
</div>
<div style="background: rgba(255,255,255,0.06); border-radius: 8px; padding: 10px 12px;
     margin-bottom: 16px; border: 1px solid rgba(255,255,255,0.1);">
  <div style="font-size: 11px; color: #64748b; margin-bottom: 4px;">PROJECT STACK</div>
  <div style="font-size: 12px; color: #94a3b8; line-height: 1.8;">
    BGE-M3 · CLIP · BM25<br>FAISS · BGE Reranker<br>LightGBM · SHAP · DeepSeek
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    '<div style="font-size:11px;color:#64748b;margin-bottom:6px;'
    'text-transform:uppercase;letter-spacing:1px;">功能模块</div>',
    unsafe_allow_html=True
)
page = st.sidebar.radio("", [
    "🤖 AI 经营助手",
    "📊 经营数据分析",
    "🎨 内容诊断",
    "🛍️ AI 选品推荐",
    "📚 运营知识库",
    "🔬 检索评估看板",
], label_visibility="collapsed")

if page == "🤖 AI 经营助手":
    from app.views.agent_chat import render
elif page == "📊 经营数据分析":
    from app.views.data_analysis import render
elif page == "🎨 内容诊断":
    from app.views.content_diagnosis import render
elif page == "🛍️ AI 选品推荐":
    from app.views.product_selection import render
elif page == "📚 运营知识库":
    from app.views.knowledge_base import render
elif page == "🔬 检索评估看板":
    from app.views.evaluation_dashboard import render

try:
    render(components)
except Exception as e:
    st.error(f"页面渲染出错：{e}")
    st.code(traceback.format_exc())
