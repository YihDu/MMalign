# **多语言视觉锚定实验全流程指南**

本文档面向开发者，汇总仓库当前（Parquet 版 COCO 多语言数据）所需的全部操作步骤，从数据加载到结果分析的完整路径。

---

## 1. 实验目标回顾

- **核心问题**：视觉信息能否作为因果锚点，强化多语言文本表征的对齐？
- **实验条件**：
  - `baseline` —— 仅文本（无视觉锚定）。
  - `correct` —— 图像与对应语言描述匹配。
  - `mismatched` —— 图像与语言描述打乱，对应错误锚定。
- **主要指标**：每个条件/语言的嵌入余弦距离；在 `analysis.statistics.t_test=true` 时，对 `baseline` vs `correct` 做独立样本 t 检验。

---

## 2. 数据来源与加载

- **数据格式**：参照 `data_guide.md`，数据位于 `/root/personal/datasets/COCO_multilingual/data`，以 Hugging Face `datasets` 的 Parquet 分片形式提供。
- **加载方式**：`data/coco_loader.py` 通过 `load_dataset("parquet", data_files=...)` 读取 `train/val/test/restval` 等 split，并自动拼接所需子集。
- **样本结构**：字段包含 `cocoid`, `filename`, `image`（PIL 对象）及多语言 caption 列（`en`, `cn`, `jp-stair`, ...，每列为最多 5 条描述）。
- **可选策略**：
  - `caption_index`：优先取某条描述（默认 0，若为空则回退到首个非空）。
  - `filter_empty_languages`：是否过滤缺少文本的样本。
  - `language_aliases`：当配置语言与列名不一致时做映射，例如 `zh -> cn`。

---

## 3. 代码架构速览

| 模块 | 作用 |
| --- | --- |
| `config/settings.yaml` | 统一管理实验参数（语言、样本数、数据 split、模型、分析/可视化选项）。 |
| `data/coco_loader.py` | 读取 Parquet 数据，构造 `SampleBatch`。 |
| `models/llava_loader.py` | 按配置加载 LLaVA 模型与 Processor。 |
| `models/embedding.py` | 将 `SampleBatch` 转成图像/文本每一层的 hidden state，并保留最终 pooled embedding 兼容旧指标。 |
| `experiment/conditions.py` | 定义 `baseline/correct/mismatched` 的样本编排逻辑。 |
| `experiment/runner.py` | 调用嵌入函数、计算余弦距离，并聚合结果。 |
| `analysis/*` | 提供余弦距离矩阵与统计检验。 |
| `visualization/plot_distance_distributions` | 输出距离分布图。 |
| `scripts/run_experiment.py` | 主脚本：串联配置 → 数据采样 → 模型推理 → 结果写入。 |
| `scripts/prepare_data.py` | 轻量检查工具：抽取一批样本并打印多语言 caption，便于确认数据有效性。 |

---

## 4. 环境准备

```bash
pip install torch transformers datasets pillow numpy scipy matplotlib pyyaml pytest
```

> 若使用 GPU，请安装匹配的 `torch` 版本。`datasets` 与 `pillow` 为 Parquet+图像读取的必备依赖。

---

## 5. 全流程操作步骤

### 5.1 配置实验

编辑 `config/settings.yaml`：

```yaml
experiment:
  sample_size: 64          # 每个条件的样本数
  languages: [en, jp-stair, cn]

data:
  coco:
    data_dir: /root/personal/datasets/COCO_multilingual/data
    splits: [train, restval]
    caption_index: 0
    filter_empty_languages: true
    language_aliases:
      cn: cn   # 如需别名，可在此声明

analysis:
  layer_mode: all          # all / indices / final
  layer_indices: []        # layer_mode=indices 时填写
  metrics:
    - cosine_distance
  statistics:
    t_test: true
```

- 若想扩展语言，只需在 `experiment.languages` 中添加字段，同时确保数据列存在。
- `sample_size` 建议在 GPU/显存允许范围内调整；`build_batch` 会按配置随机抽样。
- `analysis.layer_mode` 控制 `run_experiment` 使用哪些层做距离计算：`final`（仅 pooled）、`all`（默认，遍历每层）、`indices`（只保留 `layer_indices` 指定层）。

### 5.2 预览样本（可选）

```bash
python scripts/prepare_data.py \
  --data-dir /root/personal/datasets/COCO_multilingual/data \
  --splits train restval \
  --languages en jp-stair cn \
  --limit 5
```

该命令会打印所选语言的 caption 片段，帮助快速确认数据质量与语言可用性。

### 5.3 运行实验

```bash
python scripts/run_experiment.py --config config/settings.yaml --output report/summary.md
# 或
python main.py
```

执行流程：
1. 解析配置 → 构建 `COCODataset`，拼接所需 split。
2. 抽样 `sample_size` 个 `MultilingualExample`，确保所有语言均非空。
3. 加载 LLaVA 模型，分别编码图像与文本。
4. 针对 `baseline/correct/mismatched` 批量计算余弦距离。
5. 输出：
   - `report/summary.md`：条件 × 语言的均值统计。
   - `report/figures/{condition}.png`：距离分布图。
   - 控制台：若启用 t 检验，打印统计量与 p-value。
   - `report/distance_map.json`：详细的层级距离列表，键名形如 `cosine_en_layer_03`。

### 5.4 结果与解读

- 期望排序：`correct < baseline < mismatched`（越小越对齐）。
- 若 `baseline` 与 `correct` 的均值差异显著且通过 t 检验，则支持“视觉锚定增强对齐”的假设。
- 可结合 `analysis/statistics.py` 添加更多检验（如配对 t 检验、Bootstrap）。
- 针对多层输出，可绘制层序号 vs. 平均距离曲线，观察哪些深度对齐效果最佳；若 `layer_mode=indices`，则只关注指定层的统计。

---

## 6. 常见问题排查

| 问题 | 原因 | 解决方案 |
| --- | --- | --- |
| `ImportError: datasets` / `ImportError: PIL` | 未安装依赖 | 按“环境准备”安装。 |
| `ValueError: Unable to assemble batch...` | 指定语言数据不足 | 增大 `splits` / `sample_size`，或放宽 `filter_empty_languages`。 |
| CUDA OOM | 样本数或模型过大 | 减少 `sample_size`、使用更小模型或切换到 CPU。 |
| `FileNotFoundError: data_dir` | 配置路径错误 | 确认 Parquet 数据路径。 |
| `KeyError` for language | 语言列不存在 | 在 `language_aliases` 中加映射，或调整 `experiment.languages`。 |

---

## 7. 下一步扩展

1. **更多模型**：在 `models/llava_loader.py` 中新增配置，尝试不同规模或其他多模态模型（例如 BLIP-2）。
2. **更多条件**：在 `experiment/conditions.py` 中加入噪声文本、图像扰动等额外对照组，分析鲁棒性。
3. **指标拓展**：`analysis/metrics.py` 可新增欧氏距离、中心性衡量等，用于深化表征分析。
4. **自动化报告**：结合 `report/summary.md` 与图表生成 LaTeX/HTML 报告，便于论文写作。

---

借助以上流程，可快速在本地复现实验，验证视觉信息在多语言表征对齐中的因果影响，并为进一步研究（例如跨语言检索、视觉提示学习）打下基础。祝实验顺利！
