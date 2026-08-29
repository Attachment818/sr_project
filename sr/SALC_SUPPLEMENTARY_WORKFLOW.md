# HRF / OCTA SalC 补充实验流程

本流程用于在本地计算 SalC、筛选定性对比样例，并生成蓝色（target）、橙色（source / aligned image）、灰色（重合区域）的白底血管显著性叠加图。它不修改注册结果，也不需要 GPU。

## 数据可用性

| 数据集 | target / source | Ours aligned | 其他方法 aligned | 当前用途 |
| --- | --- | --- | --- | --- |
| HRF | 18 对 | 18 对 | 缺失 | 只能验证 Source/Ours 并生成原型图 |
| OCTA | 100 对 | 完整 | SIFT、LoFTR、NCNet、R2D2、SuperPoint、SuperRetina、RetinaRegNet 均完整 | 完整定量审计和多方法定性对比 |
| FIMD | 补充实验原始/对齐图缺失 | 缺失 | 缺失 | 暂不能生成同类图 |

## 参数来源

- HRF 使用学长 `salc.m` 的默认参数：`smoothing_radius=25`、`p=10`，绿色通道。
- OCTA 使用《对比结果表》最终“选用”参数：`smoothing_radius=1`、`p=4`，灰度图。这不是 `salc.m` 的默认参数。
- Python 实现保留了 MATLAB 的 FFT Wirtinger 滤波、百分位显著像素选择和零位移线性插值路径。

## 脚本与配置

- `tools/salc_reference.py`：SalC 参考实现、显著性叠加图及自测。
- `tools/audit_salc.py`：按 YAML 批量计算每对、每方法 SalC，输出 CSV/JSON，并按 Ours 相对最强对照的差值排序。
- `tools/render_salc_comparisons.py`：对已选 pair 生成补充材料版式，不重跑全量统计。
- `config/audit_salc_hrf_ours_v1.yaml`：HRF Source/Ours 审计。
- `config/audit_salc_octa_full_v1.yaml`：OCTA 100 对、9 组输入的全量审计。
- `config/render_salc_hrf_current_p8_v1.yaml`：使用现有数据生成 HRF pair 8 的 Source/Ours 原型图。
- `config/render_salc_octa_final_candidates_v1.yaml`：使用较强显示对比生成 OCTA pair 0045/0087 最终候选图。
- `config/render_salc_octa_candidates_v1.yaml`：OCTA 候选 pair `0045/0087/0029/0097` 的 2×5 图。

## 本地执行

```powershell
D:\Anaconda_envs\envs\sr\python.exe sr\tools\audit_salc.py --config sr\config\audit_salc_hrf_ours_v1.yaml

D:\Anaconda_envs\envs\sr\python.exe sr\tools\audit_salc.py --config sr\config\audit_salc_octa_full_v1.yaml

D:\Anaconda_envs\envs\sr\python.exe sr\tools\render_salc_comparisons.py --render-config sr\config\render_salc_octa_candidates_v1.yaml
```

每个输出目录都必须不存在或为空；脚本会拒绝覆盖旧结果。

## 当前验证结论

- OCTA：Python 复算保持了论文中的整体排序，Ours 仍最高；但部分方法与学长表格相差约 0.01–0.02，因此生成图可用于候选审阅，最终定量数字仍以论文原表为准。
- HRF：Source 复算为 0.524（论文 0.512），Ours 为 0.713（论文 0.765）。Source 较接近而 Ours 差距明显，更可能是当前 `align_image` 与论文实验版本不同；在确认数据前不应把该均值写入论文。
- OCTA 当前优先人工审阅 pair `0045`，其次是 `0087`、`0029`、`0097`。候选排序只是辅助，最终需同时排除裁剪、黑边和非物理畸变。

## 源代码校验

本流程依据的学长 MATLAB 文件 SHA-256 如下，原文件未复制入仓库、未修改：

- `wirtinger.m`: `0A6BA5D5FE9CB31B3460D4C6B06B02B419A5133A3EF77C3506C2ABADB4A63E03`
- `common_ind.m`: `C5BB6BCAC6F5945420D1E161133C6C82C00C9C9768A2681817E627C66FBB43FA`
- `imagefilter.m`: `ACD8DF97E32C8BF87DF9F415EEC0FEEFCC01F0D33A6FBD5E576D051572B7EA3C`
- `salc.m`: `44947913025A5D2F2D66867FB453020A873BE9A7C985B6AEEB22280B7CB5C0BD`
- `ShiftedLinear_Interp_2D.m`: `167DEBB7FA6A8AB1B12DD1361CCAB106EB40CE404526A26746E7F6C7F564CF9F`
- `t.m`: `13D1B4EF670DF936F9CDD36710C046AA71E89814CA35AE370C94856DD75ABAAC`
