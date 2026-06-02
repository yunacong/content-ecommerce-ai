#!/bin/bash
# 公网部署脚本：Streamlit + ngrok
# 使用前：ngrok config add-authtoken <your_token>  (在 https://dashboard.ngrok.com 获取)

cd "$(dirname "$0")"

export OMP_NUM_THREADS=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

echo "🚀 启动 Streamlit..."
streamlit run app/main.py \
  --server.fileWatcherType none \
  --server.port 8501 \
  --server.headless true &
STREAMLIT_PID=$!
echo "   Streamlit PID: $STREAMLIT_PID"

sleep 4

echo "🌐 启动 ngrok 隧道..."
echo "   访问 https://dashboard.ngrok.com/tunnels 查看公网地址"
echo "   或直接看下方输出的 Forwarding 地址"
echo ""
ngrok http 8501

# 退出时清理
kill $STREAMLIT_PID 2>/dev/null
