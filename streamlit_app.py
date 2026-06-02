"""HuggingFace Spaces 入口文件"""
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# 执行主应用
exec(compile(open(ROOT / "app" / "main.py").read(), ROOT / "app" / "main.py", "exec"))
