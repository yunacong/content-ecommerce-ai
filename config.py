from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
EVAL_DIR = DATA_DIR / "eval_sets"

# LLM
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Embedding models
BGE_MODEL = "BAAI/bge-m3"
BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAIN = "openai"

# Search params
HYBRID_TOP_K = 50       # 混合召回数量
RERANK_TOP_K = 5        # 精排后保留数量
BM25_WEIGHT = 0.3
BGE_WEIGHT = 0.7

# LightGBM
LGBM_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5, 10],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "n_estimators": 200,
    "verbose": -1,
}

# Demo data size (for fast local demo)
DEMO_PRODUCT_SIZE = 2000
DEMO_CASE_SIZE = 300
