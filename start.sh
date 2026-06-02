#!/bin/bash
# 内容电商 AI 平台启动脚本
# macOS ARM 需要 OMP_NUM_THREADS=1 避免 FAISS + LightGBM OpenMP 冲突

cd "$(dirname "$0")"

export OMP_NUM_THREADS=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

echo "🚀 启动内容电商 AI 平台..."
streamlit run app/main.py --server.fileWatcherType none "$@"
