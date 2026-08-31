# FIMD 补充图复现计划

## 当前数据边界

本地 `supplementary_experiments/fimd/` 目前只有 FIMD02 的四张已标注叠加成品：

- Target and source
- SuperRetina
- RetinaRegNet
- Ours（指扩刊论文方法，不是本项目 G0）

此外只有 RetinaRegNet 和 SuperRetina 的全数据集 MLE 文本。本地目录中没有原始 query/reference、12 对控制点、未叠加的注册图、变换矩阵或 Ours 的 `Job1_FIMD_MLE.txt`。

服务器既有测试配置指向 `/home/data1/zhangjunhong/sr_project/sr/data/FIMD`，目录结构快照记录了 `01_r_t` 到 `70_r_t`。项目的 `list_fimd_pairs()` 要求每个目录同时包含 `{id}_r.jpg`、`{id}_t.jpg` 和 `control_points_{id}_r_t.txt`；既有 FIMD 测试已经运行过，服务器上很可能已有完整原始数据，但在创建新方法命令前仍须实际执行文件计数和 FIMD02 尺寸检查。本地没有这批原始图并不等于服务器也没有。

FIMD 完整数据应包含 70 对纵向眼底图像，每对有 12 对控制点。不同时间的图像可能来自不同相机，分辨率、亮度、对比度和视野均可能不同。

## 建议方法集合

### 最小集合：与论文 FIRE Fig. 5 对齐

| 方法 | 当前状态 | 是否需要获取代码/权重 | 计算资源 | 说明 |
| --- | --- | --- | --- | --- |
| Target and source | 已有 FIMD02 成品 | 否 | CPU | 取得原图后重新统一排版 |
| SIFT | 缺失 | 不需要论文仓库，OpenCV 即可 | CPU | 必须使用与其他方法一致的变换估计和输出画布 |
| NCNet | 缺失 | 官方 `ignacio-rocco/ncnet` 与对应预训练权重 | GPU | 全分辨率相关体积可能显存过高，应按论文输入尺度运行 |
| SuperPoint | 缺失 | 先确认学长使用的实现和 checkpoint；论文相关实验常用 `rpautrat/SuperPoint` | GPU | 不同实现/权重差异明显，不能擅自换版本 |
| GeoFormer | 缺失 | 官方 `ruc-aimc-lab/GeoFormer` 与 `geoformer.ckpt` | GPU | 官方提供单对推理，但需增加 FIMD 数据适配和结果导出 |
| SuperRetina | 已有 FIMD02 成品 | 若统一重跑则使用本项目或官方 checkpoint | GPU | 本项目已有 FIMD 测试协议，可复用但必须锁定 checkpoint |
| RetinaRegNet | 已有 FIMD02 成品 | 若统一重跑则使用官方仓库 | GPU，高显存 | 官方没有现成 FIMD loader，需要适配；扩散特征开销最大 |
| Ours | 已有 FIMD02 成品 | 必须向学长获取扩刊论文代码、checkpoint 和推理配置 | GPU | 不能用 SuperRetina G0 冒充扩刊论文 Ours |

### 可选扩展集合

若老师要求 FIMD 与 HRF/OCTA 的补充图方法完全一致，再增加 LoFTR、R2D2 和 GLAMpoints。它们不是当前最小缺口，不应先占用复现时间。

## 在下载模型代码前必须向学长取得

1. 先核验服务器 FIMD 原始 70 对图像和控制点是否完整；缺失时再向学长获取，而不是重复下载。
2. FIMD02 在成品图中谁是 target、谁是 source，以及是否进行了 resize/crop。
3. 扩刊论文 Ours 的 checkpoint、输入尺寸、预处理、teacher 版本和二阶多项式估计代码。
4. Fig. 5 中 SuperPoint、NCNet、GeoFormer 的具体仓库提交、checkpoint 和参数；论文只写“默认设置并调节部分参数”，不足以精确复现。
5. 各方法最终使用原生变换模型，还是统一使用相同的 RANSAC/二阶多项式后端。

## 统一输出协议

每个方法、每个 pair 都应保存：

- 原始 target/source 路径与图像尺寸；
- 完整推理配置和代码提交；
- 匹配点、内点和变换参数；
- 对齐后的 source（不带控制点）；
- MLE/AUC 逐 pair 数值；
- 统一脚本生成的控制点叠加图。

建议先全量运行 70 对并生成逐对指标，再按“配准成功、无畸形、方法差异清楚、Ours 并非靠裁剪取胜”的规则选择定性 pair，避免只试 FIMD02 造成样本选择偏差。

## 执行顺序

1. 数据与协议预检：确认 raw images、控制点、方向和缩放规则。
2. CPU 基线：先跑 SIFT，验证坐标、MLE 和可视化链条。
3. 轻量 GPU 方法：SuperPoint、SuperRetina。
4. 匹配网络：NCNet、GeoFormer。
5. 高显存方法：RetinaRegNet。
6. 最后运行扩刊论文 Ours，并用同一评估脚本汇总。

在输入路径、checkpoint 和变换后端没有确认前，不应给出服务器 nohup 命令。
