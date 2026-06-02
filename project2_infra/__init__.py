# 懒加载：不在包级别 import 任何子模块
# 原因：faiss 和 lightgbm 在 macOS ARM 上存在 OpenMP 顺序依赖，
# 必须在业务代码里手动控制 import 顺序（lgb 先于 faiss）
