"""全局组件加载（Streamlit 缓存，只初始化一次）"""
import streamlit as st
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@st.cache_resource(show_spinner="加载 AI 模型中，首次启动需要约 1-2 分钟...")
def load_all_components():
    from config import PROCESSED_DIR, KNOWLEDGE_DIR
    from project2_infra.explainer import RecommendationExplainer
    from project1_app.text_to_sql import TextToSQL
    from project1_app.content_diagnosis import ContentDiagnosis
    from project1_app.rag_knowledge import RAGKnowledgeBase
    from project1_app.product_selection import ProductSelector
    from project1_app.agent import EcommerceAgent

    # ⚠️ macOS ARM: LightGBM 必须在 FAISS 之前加载，否则 OpenMP 冲突 segfault
    import lightgbm as lgb  # noqa: 先占 OpenMP 线程
    from project2_infra.lightgbm_ranker import LGBMRanker
    ranker = LGBMRanker()
    if ranker.is_trained():
        ranker.load()

    # FAISS 相关组件在 LightGBM 之后加载
    from project2_infra.embedder import Embedder
    from project2_infra.hybrid_search import HybridSearchEngine
    from project2_infra.reranker import BGEReranker
    from project2_infra.multimodal_retrieval import MultimodalCaseRetriever

    embedder = Embedder(use_bge=True, use_clip=False)
    reranker = BGEReranker()

    search_engine = HybridSearchEngine(index_name="products")
    if search_engine.is_built():
        search_engine.load()

    case_retriever = MultimodalCaseRetriever()
    if case_retriever.is_built():
        case_retriever.load()

    explainer = RecommendationExplainer()

    # Project 1 组件
    text_to_sql = TextToSQL()
    content_diagnosis = ContentDiagnosis(embedder=embedder, case_retriever=case_retriever)

    rag_kb = RAGKnowledgeBase(embedder=embedder, reranker=reranker)
    if rag_kb.is_built():
        rag_kb.load()

    product_selector = ProductSelector(
        hybrid_search=search_engine,
        reranker=reranker,
        ranker=ranker,
        explainer=explainer,
        embedder=embedder,
    )

    agent = EcommerceAgent(
        text_to_sql=text_to_sql,
        content_diagnosis=content_diagnosis,
        rag_kb=rag_kb,
        product_selector=product_selector,
        case_retriever=case_retriever,
        embedder=embedder,
    )

    return {
        "embedder": embedder,
        "reranker": reranker,
        "search_engine": search_engine,
        "case_retriever": case_retriever,
        "ranker": ranker,
        "explainer": explainer,
        "text_to_sql": text_to_sql,
        "content_diagnosis": content_diagnosis,
        "rag_kb": rag_kb,
        "product_selector": product_selector,
        "agent": agent,
    }
