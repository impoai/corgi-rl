# 验证状态 — 2026-10-27

本次交付边界：先搭建仓库，供用户审阅。没有提交 GPU 作业或启动 PPO 训练。

## 已执行

- 远端登录节点，在 `pytorch_291_cu128.sif` 容器中安装独立环境。
- Python 3.11.14；`uv lock` / `uv sync --locked` 完成。
- `ruff check .` 通过；stand 配置对象成功构建。此检查没有启动仿真或训练。
- `pytest tests/test_spec_and_native.py -q`：4 passed，3.85 秒。
- 生成并编译原生 MuJoCo 模型：19 个广义位置、18 个广义速度、12 个执行器。
- 固定目标 PD 控制维持站立 3 秒；这不是训练得到的站立策略。
- 本机临时环境及本次新下载的 114 个包缓存已移入废纸篓；原有 Python 未修改。

## 待运行

- `tests/test_gpu.py` 中 stand/walk 的 GPU 步进、重置和跌倒终止检查。
- mjlab 和原生 MuJoCo 的观察/动作及物理一致性。
- PPO 参数更新、有限 loss、checkpoint 保存与恢复。
- ONNX 导出、归一化、数值误差检查和规格哈希拒绝。
- EGL/H.264 视频渲染。
- 硬件控制、舵机通信、RK3576 延迟测试和任何 sim-to-real 试验。

## 已发现的兼容性问题

容器内没有 Git，GitPython 在导入 mjlab 任务时默认报错。远端 smoke 脚本
设置 `GIT_PYTHON_REFRESH=quiet`，同时要求从提交端记录源码版本。后续仍需
完整 GPU 测试确认训练路径；目前不以配置入口存在作为通过证据。

发布版 mjlab 1.6.0 使用 MuJoCo/Warp 3.11.0、rsl_rl 5.4.2；上游 main
的依赖已发生变化。安装以本仓库 `uv.lock` 为准。

## 下一轮验收

用户审阅后，先运行 GPU 测试，再分别对 stand 和 walk 做 2 次 PPO 更新。
每次记录设备、并行环境数、真实 transition 数、参数变化、loss、导出误差和
回放指标。只有这些都通过，才扩大训练；任何短程结果都不等同于收敛。
