"""Streamlit 主入口：AI 经营增长平台 Demo"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from app.components.loader import load_all_components

st.set_page_config(
    page_title="内容电商 AI 增长平台",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 全局样式 — 现代互联网风格
st.markdown("""
<style>
/* ── 全局字体与背景 ─────────────────────── */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.stApp { background: #f0f4f8; }

/* ── 侧边栏深色主题 ──────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e3a5f 100%) !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stRadio [data-checked="true"] + label {
    color: #60a5fa !important; font-weight: 600;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }
[data-testid="stSidebar"] hr { border-color: #334155 !important; }

/* ── 主内容区卡片 ──────────────────────────── */
.main-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 24px;
    margin: 12px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 4px 16px rgba(0,0,0,0.04);
    border: 1px solid #e2e8f0;
}

/* ── 指标卡片 ──────────────────────────────── */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="stMetricValue"] { color: #1e3a5f !important; font-weight: 700; }

/* ── 按钮 ──────────────────────────────────── */
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    box-shadow: 0 4px 16px rgba(37,99,235,0.4) !important;
    transform: translateY(-1px);
}
[data-testid="stButton"] button[kind="secondary"] {
    border-radius: 8px !important;
    border: 1px solid #cbd5e1 !important;
    color: #475569 !important;
}

/* ── 表格 ──────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── 输入框 ────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    border-radius: 8px !important;
    border: 1px solid #cbd5e1 !important;
    background: #f8fafc !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}

/* ── 成功/警告/错误框 ──────────────────────── */
[data-testid="stSuccess"] { border-radius: 8px; border-left: 4px solid #10b981; }
[data-testid="stWarning"] { border-radius: 8px; border-left: 4px solid #f59e0b; }
[data-testid="stError"]   { border-radius: 8px; border-left: 4px solid #ef4444; }
[data-testid="stInfo"]    { border-radius: 8px; border-left: 4px solid #2563eb; }

/* ── Expander ──────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    background: #ffffff !important;
}

/* ── 聊天气泡 ──────────────────────────────── */
[data-testid="stChatMessageContent"] {
    border-radius: 12px !important;
    border: 1px solid #e2e8f0;
}

/* ── Tab ────────────────────────────────────── */
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #2563eb !important;
    border-bottom: 2px solid #2563eb !important;
    font-weight: 600;
}

/* ── 分数高低颜色 ──────────────────────────── */
.score-high {color: #10b981; font-weight: 700;}
.score-mid  {color: #f59e0b; font-weight: 700;}
.score-low  {color: #ef4444; font-weight: 700;}

/* ── 标题区域 ──────────────────────────────── */
h1 { color: #0f172a !important; font-weight: 800 !important; letter-spacing: -0.5px; }
h2 { color: #1e3a5f !important; font-weight: 700 !important; }
h3 { color: #334155 !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# 加载全局组件（缓存）
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
<div style="background: rgba(255,255,255,0.06); border-radius: 8px; padding: 10px 12px; margin-bottom: 16px; border: 1px solid rgba(255,255,255,0.1);">
  <div style="font-size: 11px; color: #64748b; margin-bottom: 4px;">PROJECT STACK</div>
  <div style="font-size: 12px; color: #94a3b8; line-height: 1.8;">
    BGE-M3 · CLIP · BM25<br>FAISS · BGE Reranker<br>LightGBM · SHAP · DeepSeek
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div style="font-size:11px;color:#64748b;margin-bottom:6px;text-transform:uppercase;letter-spacing:1px;">功能模块</div>', unsafe_allow_html=True)
page = st.sidebar.radio("", [
    "🤖 AI 经营助手",
    "📊 经营数据分析",
    "🎨 内容诊断",
    "🛍️ AI 选品推荐",
    "📚 运营知识库",
    "🔬 检索评估看板",
], label_visibility="collapsed")

# 路由
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

render(components)
