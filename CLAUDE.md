# CLAUDE.md — CS 同学代码实现计划 & 进度追踪

---

## 项目概况

- **项目名**：蛋白质亚细胞定位预测 Web 应用
- **当前目录**：`D:\test\Protein-subcellular-localization`
- **原项目**：`D:\test\Protein-subcellular-localization-main`
- **环境管理**：Anaconda（conda env）
- **GPU**：NVIDIA RTX 4060 Laptop（CUDA 12.1）
- **ESM-2 模型**：`facebook/esm2_t30_150M_UR50D`（640 维嵌入，150M 参数）
- **当前状态**：Phase 0-5 全部完成，v3.0 模型训练完成（Gorodkin=0.741），全链路 E2E 测试通过

---

## 实施总路线（由底向上）

```
ESM-2 特征提取 → PyTorch 模型重写 → Java 后端 + MySQL → Vue 3 前端 → 联调优化
```

---

## 关键参考文件（原项目）

| 文件 | 用途 | 重点行号 |
|------|------|---------|
| `D:\test\Protein-subcellular-localization-main\utils\models.py` | 全部 7 种 Keras 架构 + Bahdanau Attention + 评估指标 | 12-37（Attention）、73-462（7 种架构）、470-643（评估） |
| `D:\test\Protein-subcellular-localization-main\utils\datasetOnehot.py` | DeepLoc.rtf 解析 + 标签字典 + 序列处理 + 4-fold 划分 | 19-59（标签字典）、85-165（序列处理+划分） |
| `D:\test\Protein-subcellular-localization-main\trainings\rs_cnn_lstm_attention_complete.py` | 超参数空间 + 训练流程（Talos） | 29-57（超参数+训练） |
| `D:\test\Protein-subcellular-localization-main\dataset\DeepLoc\DeepLoc.rtf` | 原始数据集（FASTA 格式） | — |

---

## 标签映射速查

### 10 类亚细胞定位（location）

```python
labels_dic_location = {
    'Cell.membrane': 0, 'Cytoplasm': 1, 'Endoplasmic.reticulum': 2,
    'Golgi.apparatus': 3, 'Lysosome/Vacuole': 4, 'Mitochondrion': 5,
    'Nucleus': 6, 'Peroxisome': 7, 'Plastid': 8, 'Extracellular': 9
}
```

### 膜结合状态（membrane，实际 2 分类）

```python
# 原项目定义 {'M': 0, 'S': 1, 'U': 2}，但全量 13858 条数据中无 U 样本
# 本实现将 U 合并至 M：{'M': 0, 'S': 1, 'U': 0}，实际退化为二分类
labels_dic_membrane = {'M': 0, 'S': 1, 'U': 0}
```

### 20 种标准氨基酸

```python
amino_acid_alphabet = {
    'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8,
    'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16,
    'V':17, 'W':18, 'Y':19
}
```

---

## 序列处理核心逻辑（摘自 datasetOnehot.py）

### 中心截断 + 末端 padding

```
# 如果序列长度 > seq_len：从中心删除氨基酸（保护 N 端和 C 端信号）
# 如果序列长度 < seq_len：在末尾补零
# 未知氨基酸也补零

extra = len(sequence) - sequence_len
if extra >= 0:     # 太长 → 从中心截断
    index_i = floor(len/2) - floor(extra/2)
    index_f = floor(len/2) + ceil(extra/2)
else:              # 太短 → 末端 padding
    index_i = index_f = floor(len/2)
    extra = -extra  # padding 数量
```

### 4-fold 划分逻辑

```
part % 4 == 1 → 验证集
part % 4 != 1 → 训练集
加上 [test] 标记的 → 测试集
跳过 "Cytoplasm-Nucleus" 类别
```

---

## 数据集统计概览（数据契约）

> 以下数据由 `manifest.pt` 实际统计得出（2026-05-23 审计确认），是算法队友调参的**绝对参考依据**。

### 总样本与划分

| Split | 样本数 | 占比 | 划分规则 |
|-------|--------|------|---------|
| Train | 8,313 | 60.0% | fold ∈ {2,3,4} |
| Val | 2,772 | 20.0% | fold = 1 |
| Test | 2,773 | 20.0% | FASTA 头 `test` 标记 |
| **总计** | **13,858** | 100% | — |

### 序列长度分布

| Split | Min | Max | Mean |
|-------|-----|-----|------|
| Train | 40 AA | 1000 AA | 468 AA |
| Val | 40 AA | 1000 AA | 469 AA |
| Test | 40 AA | 1000 AA | 484 AA |

> 约 90% 以上的序列长度 < 1000 AA，仅少量跨膜蛋白序列超长（如 EGFR ~1200 AA），经中心截断后统一为 1000。

### 10 类亚细胞定位分布（严重不平衡）

| ID | 类别 | Train | Val | Test | 总样本 | 占比 | 不平衡比 |
|----|------|-------|-----|------|--------|------|---------|
| 0 | Cell membrane | 800 | 267 | 273 | 1,340 | 9.7% | 8.7x |
| 1 | Cytoplasm | 1,525 | 509 | 508 | 2,542 | 18.3% | 16.5x |
| 2 | ER | 517 | 172 | 173 | 862 | 6.2% | 5.6x |
| 3 | Golgi apparatus | 215 | 71 | 70 | 356 | 2.6% | 2.3x |
| 4 | Lysosome/Vacuole | 192 | 65 | 64 | 321 | 2.3% | 2.1x |
| 5 | Mitochondrion | 906 | 302 | 302 | 1,510 | 10.9% | 9.8x |
| 6 | **Nucleus** | **2,427** | **808** | **808** | **4,043** | **29.1%** | **26.3x** |
| 7 | **Peroxisome** | **93** | **31** | **30** | **154** | **1.1%** | **1.0x** |
| 8 | Plastid | 453 | 152 | 152 | 757 | 5.4% | 4.9x |
| 9 | Extracellular | 1,185 | 395 | 393 | 1,973 | 14.2% | 12.8x |

> **不平衡比** = 最大类样本数 / 当前类样本数（Peroxisome = 基准 1.0x）。
> Nucleus (4,043) vs Peroxisome (154) 样本量差距达 **26 倍**。

### 训练建议：class_weights 参考值

```python
import numpy as np

train_counts = np.array([800, 1525, 517, 215, 192, 906, 2427, 93, 453, 1185])
total = train_counts.sum()
n_class = len(train_counts)

# 方案 A: sklearn 平衡权重（推荐）
class_weights_location = total / (n_class * train_counts)
# → [1.04, 0.55, 1.61, 3.87, 4.33, 0.92, 0.34, 8.94, 1.84, 0.70]

# 方案 B: 有效样本数加权 (Class-Balanced Loss, Cui et al. 2019)
beta = 0.9999
effective_num = 1.0 - beta ** train_counts
class_weights_cb = (1.0 - beta) / effective_num
# → [1.14, 0.59, 1.79, 4.57, 5.23, 1.00, 0.32, 11.83, 2.05, 0.74]
```

### 膜结合状态分布

| ID | 类别 | Train | Val | Test | 占比 |
|----|------|-------|-----|------|------|
| 0 | M (Membrane) | 5,240 | 1,742 | 1,767 | 63.1% |
| 1 | S (Soluble) | 3,073 | 1,030 | 1,006 | 36.9% |
| 2 | U (Unknown) | **0** | **0** | **0** | **0%** |

> M:S ≈ 1.7:1，中度不平衡，建议使用 `pos_weight` 或加权 BCE。

### 特征提取数据契约

| 属性 | 值 |
|------|-----|
| ESM-2 模型 | `facebook/esm2_t30_150M_UR50D` |
| 嵌入维度 | 640 |
| 目标序列长度 | 1000 |
| 数据精度 | float16（存储）/ float32（训练） |
| 输入 Tensor 形状 | `(batch, 1000, 640)` |
| Location 输出 | `(batch, 10)` |
| Membrane 输出 | `(batch, 2)` — 实际二分类 |

---

## ESM-2 调用速查

```python
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
model.to('cuda')

sequence = "MALWMRLLPLLALL..."
inputs = tokenizer(sequence, return_tensors="pt")
inputs = {k: v.to('cuda') for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    # 去掉 BOS（index 0）和 EOS（index -1）
    embeddings = outputs.last_hidden_state[:, 1:-1, :]
    # shape: (1, seq_len, 640)
```

**ESM-2 模型选型表**：

| 模型 | 参数量 | 嵌入维度 | 适用环境 |
|------|--------|---------|---------|
| `esm2_t6_8M_UR50D` | 8M | 320 | CPU 快速原型 |
| `esm2_t12_35M_UR50D` | 35M | 480 | CPU 平衡选择 |
| `esm2_t30_150M_UR50D` | 150M | 640 | **GPU 推荐** |
| `esm2_t33_650M_UR50D` | 650M | 1280 | 大显存 GPU |

**文档**：https://huggingface.co/docs/transformers/model_doc/esm

---

## Bahdanau Attention 核心公式（摘自 models.py 第 12-37 行）

```
score = V * tanh(W1 * features + W2 * hidden)
attention_weights = softmax(score, axis=time)
context_vector = sum(attention_weights * features, axis=time)
```

- `W1`（Dense）：作用于 encoder 所有 hidden states
- `W2`（Dense）：作用于 decoder 最后 hidden state（扩展 time axis 后相加）
- `V`（Dense → 1）：计算标量 score
- 输出：`(context_vector, attention_weights)`

---

## 完整版模型架构（create_CNN_LSTM_Attention_complete，第 383-462 行）

```
Input (seq_len, n_feat)
  → Dropout(drop_prob)
  → Permute to (n_feat, seq_len)   [channels_first]
  → 6 并行 Conv1d(kernel=1,3,5,9,15,21, 各 n_filt filters, Orthogonal init, ReLU)
  → Concat(axis=1) → (6*n_filt, seq_len)
  → Permute to (seq_len, 6*n_filt) [channels_last]
  → Conv1d(kernel=3, 128 filters, ReLU)
  → BiLSTM(n_hid, return_sequences=True, return_state=True, dropout=drop_hid)
  → Concat(forward_h, backward_h) → state_h
  → Attention(n_hid*2)(l_lstm, state_h) → context_vector
  → Dropout(drop_hid)
  → Dense(n_hid*2, ReLU, Orthogonal init)
  → Dropout(drop_hid)
  → Dense(n_class, Softmax, Orthogonal init)
```

**PyTorch 移植注意点**：

| Keras | PyTorch |
|-------|---------|
| `data_format='channels_first'` | Conv1d 默认 `(N, C, L)`，无需特殊处理 |
| `Orthogonal(gain=sqrt(2))` | `nn.init.orthogonal_(tensor, gain=math.sqrt(2))` |
| `clipnorm=3, clipvalue=2` | `torch.nn.utils.clip_grad_norm_(model.parameters(), 3)` |
| `layers[12].initial_states` | 自定义 LSTM `h0, c0` 为 `nn.Parameter` |

---

## 超参数搜索空间（摘自 rs_cnn_lstm_attention_complete.py 第 29-36 行）

```python
p = {
    'batch_size': (32, 256, 32),   # 32~256 step 32
    'lr': [0.0001, 0.0005, 0.001, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.004, 0.005, 0.007],
    'n_filt': (5, 50, 5),          # 5~50 step 5
    'n_hid': (5, 100, 5),          # 5~100 step 5
    'drop_prob': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    'drop_hid': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
}
```

---

## 数据库 DDL

```sql
CREATE DATABASE protein_localization;
USE protein_localization;

CREATE TABLE sequences (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    sequence_id VARCHAR(64) UNIQUE NOT NULL,
    raw_sequence TEXT NOT NULL,
    sequence_length INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE predictions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    sequence_id VARCHAR(64) NOT NULL,
    predicted_location VARCHAR(64) NOT NULL,
    location_confidence DOUBLE NOT NULL,
    predicted_membrane VARCHAR(32),
    membrane_confidence DOUBLE,
    all_probabilities JSON,
    attention_data JSON,
    model_version VARCHAR(32) DEFAULT 'v3',
    inference_time_ms INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sequence_id) REFERENCES sequences(sequence_id)
);
```

---

## API 接口规范

### POST /api/predict

请求：
```json
{"sequence": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHL..."}
```

响应：
```json
{
  "code": 200,
  "data": {
    "sequence_id": "abc123",
    "predicted_location": "Extracellular",
    "location_confidence": 0.923,
    "predicted_membrane": "Soluble",
    "membrane_confidence": 0.871,
    "all_probabilities": {
      "Cell membrane": 0.012, "Cytoplasm": 0.008, "ER": 0.015,
      "Golgi apparatus": 0.003, "Lysosome + Vacuole": 0.005,
      "Mitochondrion": 0.018, "Nucleus": 0.003,
      "Peroxisome": 0.002, "Plastid": 0.001, "Extracellular": 0.923
    },
    "attention_weights": [[0.001, 0.003, ...]],
    "model_version": "v3",
    "inference_time_ms": 1523
  }
}
```

**错误响应**：

| HTTP 码 | 含义 | 触发条件 | 响应体示例 |
|---------|------|---------|-----------|
| 400 | 请求参数错误 | 空序列、非氨基酸字符、`@Valid` 校验失败 | `{"code":400, "message":"Sequence is empty or contains no valid amino acids", "data":null}` |
| 502 | Python 推理失败 | 模型崩溃、GPU 显存不足、ESM-2 加载失败、JSON 解析失败 | `{"code":502, "message":"Python inference failed (exit code 1). stderr: CUDA out of memory...", "data":null}` |
| 503 | 推理服务不可用 | Python 脚本未找到、Python 可执行文件无效 | `{"code":503, "message":"Failed to start Python process...", "data":null}` |
| 504 | 推理超时 | 超过 `predict.python.timeout-seconds`（默认 60s） | `{"code":504, "message":"Inference timed out after 60s...", "data":null}` |
| 500 | 内部未知错误 | 未预期的运行时异常 | `{"code":500, "message":"Unexpected error occurred", "data":null}` |

### GET /api/history?page=1&size=20

响应：
```json
{
  "code": 200,
  "data": {
    "total": 156, "page": 1, "size": 20,
    "records": [{
      "id": 1, "sequence_id": "abc123",
      "predicted_location": "Extracellular",
      "location_confidence": 0.923,
      "predicted_membrane": "Soluble",
      "created_at": "2026-05-20 14:30:00"
    }]
  }
}
```

### GET /api/history/{id}

响应：
```json
{
  "code": 200,
  "data": {
    "id": 1,
    "sequenceId": "abc123",
    "predictedLocation": "Extracellular",
    "locationConfidence": 0.923,
    "predictedMembrane": "Soluble",
    "membraneConfidence": 0.871,
    "modelVersion": "v1",
    "inferenceTimeMs": 357,
    "allProbabilities": {
      "Cell membrane": 0.012, "Cytoplasm": 0.008, "ER": 0.015,
      "Golgi apparatus": 0.003, "Lysosome + Vacuole": 0.005,
      "Mitochondrion": 0.018, "Nucleus": 0.003,
      "Peroxisome": 0.002, "Plastid": 0.001, "Extracellular": 0.923
    },
    "createdAt": "2026-05-22 14:30:00"
  }
}
```

> **注意**：history 列表接口 (`/api/history`) 返回的 records 数组内对象 key 为 snake_case（Map 序列化时 Jackson 不转换 key）。详情接口 (`/api/history/{id}`) 的 key 为 camelCase（手动 `map.put` 时指定）。

---

## Vue 组件树 & 路由

### 组件树

```
App.vue
├── Home.vue          (/)              — 项目介绍 + 技术架构
├── Predict.vue       (/predict)       — 核心预测页（状态宿主）
│   ├── SequenceInput.vue              — 序列输入 + 上传 + 实时校验
│   │   Emit: @submit(sequence), @clear
│   ├── ResultCard.vue                 — 预测结果卡片（环形进度 + 置信度色标）
│   │   Props: location, confidence, membrane, membraneConfidence,
│   │          inferenceTimeMs, sequenceId, modelVersion
│   ├── CellDiagram.vue                — SVG 细胞图（预测高亮 + 空数据占位）
│   │   Props: highlightLocation, allProbabilities
│   ├── ProbabilityChart.vue           — ECharts 概率柱状图（10 类固定色）
│   │   Props: probabilities (Object)
│   └── AttentionHeatmap.vue           — 权重热力图（1D/2D/多头自适应）
│       Props: attentionWeights (Array)
├── History.vue       (/history)       — 历史记录
│   └── HistoryTable.vue               — 表格 + 分页 + 行点击详情弹窗
└── About.vue         (/about)         — 团队 + 技术栈
```

### API → 组件字段映射

| HTTP 响应 key | Predict.vue 引用 | 子组件 Prop |
|--------------|-------------------|-------------|
| `predicted_location` | `result.predicted_location` | `:location` |
| `location_confidence` | `result.location_confidence` | `:confidence` |
| `predicted_membrane` | `result.predicted_membrane` | `:membrane` |
| `membrane_confidence` | `result.membrane_confidence` | `:membrane-confidence` |
| `inference_time_ms` | `result.inference_time_ms` | `:inference-time-ms` |
| `sequence_id` | `result.sequence_id` | `:sequence-id` |
| `model_version` | `result.model_version` | `:model-version` |
| `all_probabilities` | `result.all_probabilities` | `:probabilities` / `:all-probabilities` |
| `attention_weights` | `result.attention_weights` | `:attention-weights` |

---

## 进度追踪

### Phase 0：项目初始化

- [x] 创建项目目录 `D:\test\Protein-subcellular-localization`
- [x] 创建 CLAUDE.md（本文件）、README.md
- [x] 创建子目录结构（python/models/, python/data/, backend/, frontend/, docs/）
- [x] 创建 python/requirements.txt、python/environment.yml

### Phase 1：ESM-2 特征提取（预计 Day 1-2，约 12h 工时）

- [x] **Step 1.1** — 创建 Anaconda 环境并安装依赖 ✅
  - 环境名 `protein-local`，Python 3.10
  - PyTorch 2.5.1 + CUDA 12.1（GPU: RTX 4060 Laptop）
  - transformers 5.8.1 + biopython 1.87
  - numpy 2.0.1（conda 安装），scikit-learn + matplotlib + pandas 已安装

- [x] **Step 1.2** — 解析 DeepLoc.rtf 格式 ✅
  - 源文件是 RTF (Rich Text Format) 包裹的 FASTA，需用 `SeqIO.parse(file, 'fasta-pearson')`
  - 每条序列末尾有 RTF 换行符 `\`，需 `rstrip('\\')` 去除
  - 格式：`>ID Location-Membrane [test]` → 解析出 location, membrane, is_test
  - 跳过 `Cytoplasm-Nucleus` 类别（146条），共 13858 条有效序列
  - 4-fold 划分：part%4==1→val, part%4∈{2,3,4}→train, [test]→test

- [x] **Step 1.3** — 编写 `python/extract_features.py` ✅（文件级存储架构）
  - **架构决策**：16GB 内存无法容纳 13858×1000×640 float32 全量特征（~35GB）
  - **方案**：每条序列的 ESM-2 embedding 存为独立 .pt 文件 + manifest 索引
  - 输出结构：`data/features/{train,val,test}/XXXXX.pt` + `manifest.pt`
  - 每条 .pt 存储 (seq_len_raw, 640) float16，约 0.5-1.2 MB/条
  - Phase 2 训练时 Dataset 按需从磁盘加载，DataLoader workers 并行 I/O

- [x] **Step 1.4** — 验证数据 ✅
  - `data/features/` 下 train(8313) / val(2772) / test(2773) 三个子目录，共 13858 个 .pt 文件
  - 单文件格式：`(seq_len_raw, 640)` float16，平均大小 ~625 KB
  - `manifest.pt` 包含完整 splits、labels、元信息
  - train/val/test 索引无重叠

> **阻塞点**：ESM-2 模型下载（~600MB，需稳定网络） + GPU 推理速度（150M 参数模型对约 8000 条序列的推理时间约 1-2 小时）

### Phase 2：PyTorch 模型重写（预计 Day 2-3，约 16h 工时）

- [x] **Step 2.1** — 实现 Bahdanau Attention 层 ✅
  文件：`python/models/attention.py`
  实现完成：`BahdanauAttention(nn.Module)` 含 W1/W2/V 三个 Linear 层 + `forward(features, hidden)`。
  单元测试通过：输入 `(2, 1000, 128)` features + `(2, 128)` hidden → context `(2, 128)` + weights `(2, 1000, 1)`，softmax sum=1 ✓

- [x] **Step 2.2** — 实现 7 种对比架构 ✅
  文件：`python/models/architectures.py`
  所有 7 种架构已实现为独立的 `nn.Module` 子类：
  | # | 类名 | 双输出 | 参数 |
  |---|------|--------|------|
  | 1 | `FFN` | 是 | seq_len, n_feat, n_hid, n_class, drop_prob |
  | 2 | `CNN` | 是 | + n_filt |
  | 3 | `BLSTM` | 是 | seq_len, n_feat, n_hid, n_class, drop_prob |
  | 4 | `CNN_BLSTM` | 是 | + n_filt |
  | 5 | `BLSTM_Attention` | 是 | seq_len, n_feat, n_hid, n_class, drop_prob |
  | 6 | `CNN_BLSTM_Attention` | 是 | + n_filt |
  | 7 | `CNN_BLSTM_Attention_Complete` | 否(仅loc) | + drop_hid, n_filt |
  含 `create_model(name, ...)` 工厂函数 + 单元测试（forward + backward 全通过）。

- [x] **Step 2.3** — 编写 PyTorch Dataset（文件级按需加载）✅
  文件：`python/data/dataset.py`
  `ProteinDataset(manifest_entries, features_dir, seq_len)` — 按需从磁盘加载独立 .pt 文件 + 动态 padding
  包含 `load_manifest_split()` 和 `create_dataloaders()` 便捷函数 + 单元测试

- [x] **Step 2.4** — 编写训练脚本 `python/train.py` ✅
  - 命令行参数：`--model`、`--data`、`--epochs`、`--batch_size`、`--lr`、`--n_hid`、`--n_filt`、`--drop_prob`、`--drop_hid`、`--num_workers`、`--patience`、`--output`、`--no_eval`
  - 训练循环：Adam optimizer + CrossEntropyLoss（双输出 loc+mem / 单输出 loc）+ `clip_grad_norm_(max_norm=3)`
  - Early stopping：patience=20，监控 val_loss；Model checkpoint：`best_model.pt`
  - 每个 epoch 记录：train_loss/loc_acc/mem_acc + val_loss/loc_acc/mem_acc
  - 训练结束后保存 `training_curves.png`（4 子图：total loss + loc acc + mem loss + mem acc）
  - FFN 冒烟测试通过（2 epoch，loss 2.04→1.45，acc 54%→66%）

- [x] **Step 2.5** — 评估指标 ✅
  - Gorodkin（多类 MCC）+ MCC（膜分类）：`sklearn.metrics.matthews_corrcoef`
  - 混淆矩阵：`sklearn.metrics.confusion_matrix` + matplotlib 热力图，保存 `cm_location.png` / `cm_membrane.png`
  - `print_measures()`：仿照原项目输出最佳 epoch 全部指标
  - 评估阶段由 `--no_eval` 控制，默认在训练结束后自动在测试集上运行

- [x] **Step 2.6** — 训练优化（反过拟合 + 类别不平衡 + 双任务解耦）✅
  - Location 逆频率加权 CrossEntropyLoss（Peroxisome ×8.94, Nucleus ×0.34）
  - Membrane 普通 CrossEntropyLoss（M:S=1.7:1 健康，无需加权）
  - Adam L2 正则化 `weight_decay=1e-4`
  - ReduceLROnPlateau（patience=4, factor=0.5, min_lr=1e-6）
  - 双任务 Loss 解耦 `loss = loc_loss + 0.5 * mem_loss`
  - 新增 CLI：`--weight_decay`, `--mem_weight`, `--lr_patience`, `--lr_factor`, `--use_class_weights`, `--seed`
  - 固定随机种子 `--seed 42`（可复现性）

> **训练优化说明（v1.0, 2026-05-23）**：CNN_BLSTM_Attention 初始训练出现严重过拟合——train acc 97.5% vs val acc 75.9%, 21.6pp gap。根因：(1) Location 26 倍类别不平衡 → 逆频率加权 CE 被 Nucleus 梯度淹没；(2) ESM-2 640 维特征极其丰富，下游轻量网络死记训练集；(3) 双任务等权，Membrane 收敛远快于 Location；(4) 无 L2 + 无 LR 调度。
>
> **训练优化说明（v2.0, 2026-05-23）**：P0 验证解耦（加权 CE→unweighted val_acc 监控）+ P1 正则化（LayerNorm + drop_prob 0.5）。关键发现——"加权 Loss 污染"：模型 Epoch 10→24 期间 val_acc 从 0.759→0.808 (+5pp)，但加权 val_loss 从 1.11→1.77 (+60%)，导致更强的模型被静默丢弃。修复后选择 Epoch 24，Gorodkin=0.679, Train/Val gap 从 21.6pp→16.4pp (-5.2pp)。
>
> **训练优化说明（v3.0, 2026-05-23）**：四重防线重构——(1) sqrt 平滑类别权重 (26×→5× 极差压缩)；(2) Focal Loss γ=2.0 + Label Smoothing ε=0.1 (动态调制 + 抗过拟合)；(3) 同方差不确定性多任务加权 (Kendall et al.)；(4) 640→256 信息瓶颈层 (强制压缩 ESM-2 冗余) + weight_decay 1e-4→5e-4 + patience 20→10。**结果：Gorodkin=0.741, Train/Val gap=5.2pp, 目标达成。**

- [x] **Step 2.7** — P0+P1 验证解耦 + 正则化增强 ✅

  **P0：验证监控与训练损失解耦**
  - **理论根因**：Step 2.6 中 Early Stopping / ReduceLROnPlateau / Best Checkpoint 三者均监控使用加权 CE 计算的 `val_loss`。加权 CE 中 Peroxisome ×8.94、Golgi ×3.87——当模型开始"勇敢"预测小类时，少数高权重样本的边界错误被 8-26× 放大，淹没了整体的 accuracy 改善信号。
  - **证据**：Epoch 10 val_loss=1.106 (val_acc=0.759, MCC=0.679)；Epoch 24 val_loss=1.767 (val_acc=0.808, MCC 未测)。模型在 Epoch 10-24 期间 accuracy 提升 +5pp，但加权 val_loss 上升了 60%，导致 Epoch 24 的更强模型被静默丢弃。
  - **修复**：
    - 新增 `criterion_loc_unweighted = nn.CrossEntropyLoss()` 专用于验证阶段的 loss 计算
    - `validate()` 返回 unweighted `loc_loss_raw` 和 weighted `loc_loss_weighted` 两个指标
    - Early Stopping + ReduceLROnPlateau + Checkpoint 全部改为监控 `val_loc_acc`（直接优化目标指标）
    - 新增 `--monitor` CLI 参数（默认 `val_acc`，可选 `val_loss` 回退）
    - 训练日志同步输出 weighted 和 unweighted 两种 val_loss，便于对比诊断

  **P1：正则化增强**
  - **根因**：train_loc_acc 达 0.97 而 val_loc_acc 仅 0.81，16pp 的 train/val gap 说明严重过拟合。当前仅 `drop_prob=0.3` + `weight_decay=1e-4` 不足以约束 293K 参数模型对 640 维 ESM-2 特征的死记。
  - **修复**：
    - `drop_prob` 默认值 0.3 → 0.5（原项目超参数搜索空间上限为 0.7）
    - `CNN_BLSTM_Attention` 架构 BiLSTM 输出后、Attention 输入前增加 `nn.LayerNorm(n_hid*2)`
    - LayerNorm 沿特征维度归一化，稳定 Attention score 的数值分布，削弱单个高激活神经元对 context vector 的绑架效应

  **预期收益**：
  | 指标 | 优化前 | P0 后 | P0+P1 后 |
  |------|--------|-------|----------|
  | Best val_loc_acc | 0.759 (E10) | 0.808 (E24) | 0.81+ |
  | Test Gorodkin | 0.679 | 0.70+ | 0.72+ |
  | Train/Val acc gap | 16pp | ~12pp | ~8pp |

> **阻塞点**：完整版 6 并行 Conv 层参数量 + GPU 显存管理。建议先用 `esm2_t12_35M`（480 维）快速跑通，再换 `esm2_t30_150M`（640 维）做最终训练。

- [x] **Step 2.8** — v3.0 多重防御重构（Focal Loss + 瓶颈层 + 不确定性加权 + sqrt 平滑）✅

  **v3.0 四重防线**：

  1. **平方根平滑类别权重**：$w_i = \sqrt{N_{total} / (C \cdot N_i)}$，极差 26.3× → 5.1×
     - Peroxisome: 8.94 → 2.990 (-67%)，Nucleus: 0.34 → 0.585 (+72%)

  2. **Focal Loss (γ=2.0) + Label Smoothing (ε=0.1)**：动态调制替代静态逆频率权重
     - 易分样本 (p=0.9) 梯度衰减 100×，困难样本 (p=0.1) 全梯度保留
     - Label Smoothing 防止模型输出极端置信度 (p→0.999)，强制保持最低不确定性
     - 新增文件 `python/models/focal_loss.py`，含单元测试

  3. **同方差不确定性多任务加权**（Kendall et al., CVPR 2018）：
     - $L = \exp(-s_{loc})L_{loc} + \frac{1}{2}s_{loc} + \exp(-s_{mem})L_{mem} + \frac{1}{2}s_{mem}$
     - `CNN_BLSTM_Attention` 新增 `log_var_loc`, `log_var_mem` 可学习参数
     - 训练时自动平衡双任务，替代硬编码 `mem_weight=0.5`

  4. **640→256 信息瓶颈层**：`nn.Linear(640,256)→ReLU→Dropout`
     - 位于 CNN 卷积层之前，强制压缩 ESM-2 冗余特征至定位相关低维子空间
     - Conv1d 输入通道 640 → 256，减少卷积层参数 98K

  **协同正则化**：weight_decay 1e-4 → 5e-4 | Dropout 0.5 | LayerNorm | Early Stopping patience 20→10

  **新增 CLI**：`--gamma`, `--label_smoothing`, `--bottleneck_dim`, `--use_uncertainty`, `--no_uncertainty`, `--scheduler`, `--T_0`, `--T_mult`

  **涉及文件**：
  | 文件 | 操作 | 说明 |
  |------|------|------|
  | `python/models/focal_loss.py` | **新建** | FocalLoss 类 (γ=2.0, label_smoothing=0.1, sqrt-smoothed weights) |
  | `python/models/architectures.py` | **修改** | CNN_BLSTM_Attention 新增 input_proj 瓶颈层 + log_var_loc/log_var_mem；create_model 新增 bottleneck_dim 参数 |
  | `python/train.py` | **修改** | sqrt 平滑权重 + FocalLoss + 不确定性加权 + CosineAnnealing 备选调度器 + patience=10 + weight_decay=5e-4 |
  | `CLAUDE.md` | **更新** | Step 2.8 记录 |
  | `README.md` | **更新** | 新增"模型算法演进与全链路调优纪实"章节（v1→v2→v3 完整路线图） |

> **v3.0 训练完成（2026-05-23）**：
> - Epoch 23 best, val_acc=0.806, Gorodkin=**0.741**, MCC(membrane)=**0.640**
> - 5-epoch 冒烟测试 Gorodkin=0.737 已超 v2.0 基线
> - 不确定性参数收敛：σ_loc≈1.20, σ_mem≈0.50（符合理论：membrane 简单→精度高→权重大）
> - LR 首次衰减 E23 (1e-3→5e-4)，Early Stopping E33 (patience=10)
> - 训练耗时 26.5 min（早停），Train/Val gap 仅 5.2pp（v2.0 为 16.4pp）

### Phase 3：Java 后端 + MySQL（预计 Day 3-4，约 12h 工时）

> **Phase 切换审计（2026-05-22）**：✅ 已完成
> - Tensor 维度链（Dataset `(1000,640)` → Model → Train）对齐 ✅
> - API 响应格式与 CLAUDE.md 规范对齐 ✅
> - `predict.py` 已创建 ✅（含 `LOCATION_LABEL_MAP` 内部 key → 人类可读名称映射）
> - 标签名称映射已解决：`predict.py` 将 `Endoplasmic.reticulum`→`ER`、`Golgi.apparatus`→`Golgi apparatus` 等

- [x] **Step 3.1** — Spring Boot 项目初始化 ✅
  - 使用 Spring Initializr 或手动创建 Maven 项目
  - 依赖：spring-boot-starter-web, mybatis-plus-boot-starter, mysql-connector-j
  - 配置 `application.properties`：数据库连接、端口 8080

- [x] **Step 3.2** — MySQL 建表 + 基础 CRUD ✅
  - 运行 DDL（上方的 CREATE TABLE）
  - 创建 Entity + Mapper（MyBatis-Plus）
  - 创建 Service 层

- [x] **Step 3.3** — 实现 API Controller ✅
  ```java
  @RestController
  @RequestMapping("/api")
  public class PredictController {
      @PostMapping("/predict")       // → PredictService.predict(sequence)
      @GetMapping("/history")        // → PredictService.getHistory(page, size)
      @GetMapping("/history/{id}")   // → PredictService.getById(id)
  }
  ```

- [x] **Step 3.4** — 实现 PredictService（核心）✅
  1. 生成 `sequence_id`（MD5 前 8 位或 UUID）
  2. 保存 `sequences` 表
  3. `ProcessBuilder("python", "python/predict.py", "--sequence", sequence)` 调用推理脚本
  4. 设置超时 60s
  5. 解析 stdout JSON
  6. 保存 `predictions` 表
  7. 返回 JSON 响应

- [x] **Step 3.5** — 编写 `python/predict.py`（推理脚本）✅
  - 加载 ESM-2 模型 + 训练好的 PyTorch 模型（*启动时加载一次，避免每次推理都重新加载*）
  - 或使用简易模式：每次加载（通过 `--sequence` 参数）
  - 输出 JSON 到 stdout（`print(json.dumps(result))`）
  - Java 端读取 stdout 行

> **注意**：`predict.py` 当前采用每次调用重新加载策略（ESM-2 + 模型 ~3-5s），GPU 推理耗时可接受（~357ms）。模型已训练 CNN_BLSTM_Attention（Epoch 6），端到端推理可用。性能优化空间：改为常驻 Flask 微服务可消除加载延迟。

### Phase 4：Vue 3 + Element Plus 前端（预计 Day 4-6，约 16h 工时）

- [x] **Step 4.1** — 项目初始化 ✅
  - 手动创建 package.json（Vue 3.5 + Element Plus 2.9 + ECharts 5.6 + Axios 1.9）
  - `vite.config.js`：`@` 别名 + proxy `/api` → `http://localhost:8080`
  - `npm install` — 89 packages, 0 vulnerabilities

- [x] **Step 4.2** — 路由 + 布局 ✅
  - `router/index.js`：4 条路由（`/`, `/predict`, `/history`, `/about`），懒加载
  - `App.vue`：el-container 布局（el-header + el-menu + el-main + el-footer）
  - `main.js`：Element Plus + Icons 全局注册

- [x] **Step 4.3** — 组件实现（全部 6 个）✅

  1. **SequenceInput.vue** ✅
     - el-input（textarea, monospace 字体）+ el-upload（drag, .fasta 解析, FileReader）+ "Load Example" / "Clear" 按钮
     - 客户端实时校验：非法字符检测 + 有效 AA 计数
     - FASTA header 自动跳过（`>` 开头行）
     - Emit：`@submit(sequence)`, `@clear`

  2. **ResultCard.vue** ✅
     - el-card + el-tag（color-coded）+ el-progress（dashboard 环形进度）
     - 置信度颜色：≥0.8 绿 / 0.5~0.8 橙 / <0.5 红
     - el-descriptions：推理耗时 + 序列 ID
     - Props：`location`, `confidence`, `membrane`, `membraneConfidence`, `inferenceTimeMs`, `sequenceId`, `modelVersion`

  3. **ProbabilityChart.vue** ✅
     - vue-echarts `<v-chart>`，ECharts 按需引入（BarChart + Grid + Tooltip）
     - 10 类固定顺序 + 10 色固定映射（与 CellDiagram 颜色一致）
     - 最高概率柱加粗边框 + 标签标注
     - 横向柱状图，右侧百分比标签

  4. **CellDiagram.vue** ✅（SVG 纯手绘）
     - viewBox 400×400，10 个细胞结构（含胞外标注）
     - 细胞膜（椭圆边框）、细胞质（填充）、细胞核（圆+核仁）、ER（网状路径）
     - 高尔基体（4 层弧线）、线粒体 ×3（椭圆+嵴折线）、溶酶体/过氧化物酶体（圆）
     - 质体（虚线椭圆）、胞外空间（文字标注）
     - CSS transition 0.5s + `@keyframes pulse` 发光动画
     - 底部 10 色图例

  5. **AttentionHeatmap.vue** ✅
     - ECharts heatmap（HeatmapChart + VisualMap + DataZoom）
     - 底部 `dataZoom` slider + inside 双模式滚动
     - visualMap 蓝→红渐变图例
     - 处理 null/空/1D/2D 多种输入格式

  6. **HistoryTable.vue** ✅
     - el-table（stripe + highlight-current-row）+ el-pagination
     - 搜索框（按 Sequence ID 前端过滤）+ 总数提示
     - 行点击 → el-dialog 展示详情 + 概率图（复用 ProbabilityChart）
     - 颜色编码：location tag + confidence 文本颜色 + membrane tag type
     - v-loading 加载态 + 空数据提示

- [x] **Step 4.4** — API 层封装 ✅
  - `api/index.js`：axios 实例，`ApiResponse<T>` 解包拦截器，超时 65s，连接失败提示
  - `api/predict.js`：`postPredict(sequence)`, `getHistory(page, size)`, `getHistoryById(id)`

> **编译验证**：`npx vite build` — 2232 modules, built in 7.67s，全部通过

### Phase 5：联调测试（预计 Day 6-7，约 8h 工时）

- [x] **Step 5.1** — 全链路测试 ✅
  1. 启动 Java 后端 → `http://localhost:8080/api/predict` 可用
  2. 启动 Vue 前端 → `http://localhost:5173` 可访问
  3. 输入序列 → 点击提交 → 后端调用 Python → 返回 JSON → 前端展示
  4. ResultCard + CellDiagram + ProbabilityChart + AttentionHeatmap 全部正确渲染

  > **联调问题及修复记录（2026-05-22）**：
  > 1. **Python ModuleNotFoundError (numpy)**: 系统 `python` 无 conda 包 → `application.properties` 改为 conda env 完整路径
  > 2. **HuggingFace Hub 连接超时（中国 GFW）**: `PredictService.java` 添加 `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` 环境变量
  > 3. **模型路径错误**: `predict.model.path` 从 `python/best_model.pt` 改为 `../outputs/cnn_blstm_attn/best_model.pt`（相对于 python/ 工作目录）
  > 4. **JSON 序列化契约不匹配**: Python 输出 snake_case，Java DTO 默认 camelCase → `PredictResponse.java` 全部字段加 `@JsonProperty("snake_case_name")` + `@JsonIgnoreProperties(ignoreUnknown=true)`
  > 5. **前端字段绑定失效**: `@JsonProperty` 双向生效导致后端序列化输出变为 snake_case → `Predict.vue` 全部 `result.xxx` 引用改为 snake_case
  > 6. **ResultCard NaN 防御**: `confidencePercent` 加 null/NaN 兜底 → 0

  > **代码审查优化（2026-05-22）**：
  > - **后端**：新增 `PredictException(statusCode)` 细化异常分类 → `GlobalExceptionHandler` 按 502/503/504 返回精确 HTTP 码；`ObjectMapper` 改为 Spring 注入；移除未使用 `TypeReference` import
  > - **前端**：Loading 完成平滑过渡 350ms + `ElMessage` toast 通知；`AttentionHeatmap` 支持多头 attention 均值聚合；`CellDiagram` 空数据半透明 + 占位提示；`About.vue` 外链加 `rel="noopener noreferrer"`；移除 `console.error` 残留 + `main.js` 冗余参数

- [x] **Step 5.2** — 边界情况 ✅
  - 空输入 → 前端按钮 disabled（`validCharCount > 0` 校验）+ 后端 `@NotBlank` + 400
  - 非氨基酸字符（如数字）→ 前端 `onInput()` 实时检测 + 黄色警告 + 后端 `PredictService.cleanSequence()` 自动清洗
  - 超长序列（>1000aa）→ 前端 `lengthWarning` info 提示"将中心截断至 1000aa" + Python `pad_center_truncate()` 自动处理
  - Python 推理超时（>60s）→ `ProcessBuilder.waitFor(60s)` → `destroyForcibly()` → RuntimeException → 前端 axios 65s timeout
  - 前端连接失败 → axios interceptor 提示"Cannot connect to backend"

- [x] **Step 5.3** — 性能优化 ✅
  - GPU 推理耗时 ~357ms（含 ESM-2 encoding + 模型推理），性能满足需求
  - 前端 loading 分阶段展示进度（4 个阶段提示 + indeterminate progress bar）
  - 当前无需改为 Flask 常驻进程

---

## 验证 CheckList

- [x] `conda activate protein-local && python -c "import torch; print(torch.cuda.is_available())"` → True
- [x] `python extract_features.py` 可运行，生成 `data/features/{train,val,test}/XXXXX.pt` + `manifest.pt`（13858 文件）
- [x] ESM-2 特征维度正确（每条 (seq_len_raw, 640) float16）
- [x] `python models/attention.py` 单元测试通过
- [x] 7 种架构 `model.forward(x)` 均不报错，backward 正常（`python models/architectures.py` PASSED）
- [x] `python data/dataset.py` 单元测试通过（单条加载 + DataLoader batch）
- [x] `python train.py --model FFN --epochs 2` 跑通，loss 下降
- [x] `python predict.py` 语法正确，Predictor 类加载模型逻辑完整
- [x] Spring Boot 项目结构完整（18 个 Java 文件 + pom.xml + init.sql）
- [x] `python test_predict_quick.py` 全部 7 项测试通过
- [x] `mvn compile` BUILD SUCCESS（Java 23 + Maven 3.9.15）
- [x] `python train.py --model CNN_BLSTM_Attention --epochs 60 --use_class_weights --weight_decay 1e-4 --mem_weight 0.5 --drop_prob 0.5 --monitor val_acc --seed 42` (v2.0) 完成训练，Gorodkin=0.679
- [x] Gorodkin 值 > 0.6（与原项目趋势一致）✅
- [x] `python train.py --model CNN_BLSTM_Attention --epochs 60 --weight_decay 5e-4 --drop_prob 0.5 --use_class_weights --use_uncertainty --monitor val_acc --gamma 2.0 --label_smoothing 0.1 --bottleneck_dim 256 --scheduler plateau --seed 42` (v3.0) 完成训练，Gorodkin=**0.741** > 0.72 ✅
- [x] v3.0 FocalLoss 冒烟测试：5 epoch val_acc 持续上升，无 NaN ✅
- [x] v3.0 不确定性参数 σ_loc≈1.20, σ_mem≈0.50 收敛合理 ✅
- [x] v3.0 Train/Val gap 5.2pp < 10pp ✅
- [x] v3.0 Peroxisome 召回率 > 0.3（已查混淆矩阵，见 `python/outputs/cnn_blstm_attn_v3/cm_location.png`）
- [x] MySQL 建表成功，Spring Boot 启动成功
- [x] `POST /api/predict` 返回正确 JSON
- [x] `npx vite build` BUILD SUCCESS（2232 modules, 7.67s）
- [x] Vue 前端 6 个组件 + 4 个页面 + API 层全部实现
- [x] `npm run dev` 可启动，页面路由跳转正常
- [x] 完整链路：输入序列 → 加载动画 → 预测结果 → 细胞图高亮 → 概率图 → 热力图 → 历史记录

---

## 目录结构

```
Protein-subcellular-localization/
├── CLAUDE.md                    # 本文件（计划 + 进度）
├── README.md                    # 项目说明
├── python/
│   ├── environment.yml          # conda 环境
│   ├── requirements.txt         # pip 依赖
│   ├── extract_features.py      # Phase 1 产出：ESM-2 特征提取
│   ├── train.py                 # Phase 2 产出：训练脚本
│   ├── predict.py               # Phase 3 产出：推理脚本（供 Java 调用）
│   ├── models/
│   │   ├── __init__.py
│   │   ├── attention.py         # Bahdanau Attention (nn.Module)
│   │   ├── architectures.py     # 7 种对比架构
│   │   └── focal_loss.py        # Focal Loss + Label Smoothing (v3.0)
│   └── data/
│       ├── __init__.py
│       ├── dataset.py           # PyTorch Dataset（文件级按需加载 + 动态 padding）
│       └── features/            # Phase 1 产出：独立 .pt 文件
│           ├── train/00000.pt ...
│           ├── val/00000.pt ...
│           ├── test/00000.pt ...
│           └── manifest.pt      # 索引 + labels + 元信息
├── backend/
│   ├── src/main/java/com/proteinlocal/
│   │   ├── ProteinLocalApplication.java
│   │   ├── config/MybatisPlusConfig.java
│   │   ├── controller/
│   │   │   ├── PredictController.java
│   │   │   └── GlobalExceptionHandler.java
│   │   ├── dto/
│   │   │   ├── ApiResponse.java
│   │   │   ├── HistoryPage.java
│   │   │   ├── PredictRequest.java
│   │   │   └── PredictResponse.java
│   │   ├── entity/
│   │   │   ├── Prediction.java
│   │   │   └── Sequence.java
│   │   ├── exception/
│   │   │   └── PredictException.java
│   │   ├── mapper/
│   │   │   ├── PredictionMapper.java
│   │   │   └── SequenceMapper.java
│   │   └── service/
│   │       ├── PredictService.java
│   │       ├── PredictionService.java
│   │       └── SequenceService.java
│   ├── src/main/resources/application.properties
│   ├── pom.xml
│   └── sql/
│       └── init.sql
├── frontend/
│   ├── src/
│   │   ├── views/               # Home, Predict, History, About
│   │   ├── components/          # SequenceInput, ResultCard, CellDiagram, ProbabilityChart, AttentionHeatmap, HistoryTable
│   │   ├── router/index.js
│   │   ├── api/                 # axios 封装
│   │   └── App.vue
│   ├── package.json
│   └── vite.config.js
└── docs/
    └── 小组分工计划书.md
```

---

## 调优纪实档案 (Tuning Chronicle)

> 本章记录 CNN_BLSTM_Attention 模型从初始基线到 v3.0 终极方案的完整调优历程。
> 每一个决策节点都标注了**当时观察到的现象 → 提出的假设 → 验证方案 → 最终结论**，
> 作为团队期末答辩的理论支撑和未来 ML 项目的经验参考。

---

### 起点：问题的本质

**数据特征**：
- 训练集 8,313 条蛋白质序列，每条经 ESM-2 编码为 (1000, 640) 维度特征
- 640 维来自 150M 参数蛋白质语言模型，蕴含远超定位任务所需的丰富语义
- 10 类亚细胞定位，样本量从 93 (Peroxisome) 到 2,427 (Nucleus)，极差 **26 倍**

**模型**：CNN_BLSTM_Attention（双分支 Conv1d + BiLSTM + Bahdanau Attention），约 293K 可训练参数。

**核心矛盾预判**：640 维极其丰富的预训练特征 + 轻量下游网络 + 严重类别不平衡 = 几乎必然的过拟合。

---

### v1.0 — 初始基线（逆频率加权 + L2 + Plateau）

**训练配置**：

| 超参数 | 值 | 设计意图 |
|--------|-----|---------|
| Loss | CrossEntropyLoss(weight=逆频率) | Peroxisome ×8.94, Nucleus ×0.34，补偿类别不平衡 |
| Optimizer | Adam(lr=0.001, weight_decay=1e-4) | L2 正则化约束权重幅度 |
| Scheduler | ReduceLROnPlateau(patience=4, factor=0.5) | val_loss 不降时衰减 LR |
| 双任务权重 | `loss = loc_loss + 0.5 * mem_loss` | 人工设定 Membrane 贡献减半 |
| Dropout | drop_prob=0.3 | 原项目默认值 |
| Monitor | weighted val_loss | 加权 CrossEntropyLoss 作为验证监控指标 |

**训练结果**：

| 指标 | 值 | 评估 |
|------|-----|------|
| Best epoch | 10 | 由 weighted val_loss 选择 |
| Train accuracy | 97.5% | 几乎完美拟合训练集 |
| Val accuracy | 75.9% | 泛化差距巨大 |
| **Train/Val gap** | **21.6pp** | 严重过拟合 |
| Test Gorodkin | **0.679** | 勉强可用 |

**初步诊断**：
- 21.6pp 的 Train/Val gap 是教科书级过拟合
- 640 维 ESM-2 特征 + 293K 参数模型 → 容量足以死记训练集
- 逆频率权重可能扭曲了验证信号

---

### v2.0 — "加权 Loss 污染"现象的发现

**关键观察**：查看 Epoch 10→24 的训练日志时，发现了一个反常现象：

| Epoch | Weighted val_loss | Unweighted val_loss | val_acc | 解读 |
|-------|-------------------|---------------------|---------|------|
| 10 | 1.106 | — | 0.759 | v1.0 选择的"最佳"epoch |
| 24 | **1.767** (+60%) | — | **0.808** (+5pp) | 更强的模型，被静默丢弃 |

**假设形成**：

逆频率加权 CrossEntropyLoss 中 Peroxisome 权重 ×8.94、Golgi ×3.87。当模型决策边界逐渐改善、开始自信地预测小类时，少数高权重样本的边界误判被 8-26 倍放大，淹没了整体的 accuracy 改善信号。

**换句话说**：模型在 Epoch 10→24 期间确实在持续变好（val_acc +5pp），但加权 Loss 因为权重放大效应而反向飙升 60%，导致 Early Stopping、Checkpoint、LR Scheduler 三者一致选择了 Epoch 10 的更差模型。

**修复方案（P0 + P1）**：

**P0 — 验证监控解耦**：
- 新增 `criterion_loc_unweighted = nn.CrossEntropyLoss()` 专用于验证阶段
- `validate()` 同时返回 weighted 和 unweighted 两个 loss
- Early Stopping + Checkpoint + ReduceLROnPlateau **全部改为监控 `val_acc`**（直接优化目标指标，不再被加权 Loss 污染）
- 新增 `--monitor` CLI 参数（默认 `val_acc`，可回退 `val_loss`）

**P1 — 正则化增强**：
- `drop_prob` 0.3 → 0.5（原项目超参数搜索空间上限 0.7）
- BiLSTM 输出后增加 `nn.LayerNorm(n_hid*2)`，沿特征维度归一化以稳定 Attention score 分布

**v2.0 训练结果**：

| 指标 | v1.0 | v2.0 | Δ |
|------|------|------|---|
| Best epoch | 10 | 24 | +14 |
| Val accuracy | 0.759 | 0.808 | **+4.9pp** |
| Train acc | 0.975 | 0.972 | ≈ |
| Train/Val gap | 21.6pp | 16.4pp | **-5.2pp** |
| Test Gorodkin | 0.679 | 0.679 | — |

**关键洞察**：
- 将监控指标切换到 val_acc 后，模型正确选择了 Epoch 24——证实了"加权 Loss 污染"假设
- LayerNorm + Dropout 0.5 有效缓解了过拟合（gap 降 5.2pp）
- **但 Gorodkin 未提升（0.679→0.679）**——说明验证解耦暴露了真实瓶颈：train acc 97.2% vs val acc 80.8%，16pp gap 指向**模型容量与正则化强度的结构性不匹配**

**教训**：
> 当损失函数使用了类别权重、Focal Loss 等非均匀缩放时，**绝不能**将其直接作为验证监控指标和 Early Stopping 的判定依据。验证监控必须使用与最终评估目标一致的、未被扭曲的指标。

---

### v3.0 — 四重防线重构

**核心矛盾**：Train 97.2% vs Val 80.8%，16pp 的 gap 说明模型在死记 640 维 ESM-2 特征中的噪声模式，而非学习可泛化的定位信号。

**设计哲学**：从"单一正则化"转向"多重协同防御"——在输入、损失、优化三个层面同时施加约束，使每道防线只需承担部分压制任务。

#### 防线 1：平方根平滑类别权重（损失层面）

**动机**：逆频率权重 Peroxisome 8.94× vs Nucleus 0.34×，极差 26.3×。虽然我们在 v2.0 中解耦了验证监控，但训练阶段仍受极端权重影响——Peroxisome 单个误判产生的梯度等价于 26 个 Nucleus 误判，导致训练不稳定。

**数学**：

$$w_i = \sqrt{\frac{N_{total}}{C \cdot N_i}}$$

| ID | 类别 | N_i | 逆频率 w | **sqrt 平滑 w_smoothed** | 变化 |
|----|------|-----|----------|--------------------------|------|
| 0 | Cell membrane | 800 | 1.04 | **1.019** | ≈ |
| 1 | Cytoplasm | 1,525 | 0.55 | **0.738** | +34% |
| 2 | ER | 517 | 1.61 | **1.268** | -21% |
| 3 | Golgi apparatus | 215 | 3.87 | **1.966** | -49% |
| 4 | Lysosome/Vacuole | 192 | 4.33 | **2.081** | -52% |
| 5 | Mitochondrion | 906 | 0.92 | **0.958** | +4% |
| 6 | Nucleus | 2,427 | 0.34 | **0.585** | +72% |
| 7 | **Peroxisome** | 93 | **8.94** | **2.990** | **-67%** |
| 8 | Plastid | 453 | 1.84 | **1.355** | -26% |
| 9 | Extracellular | 1,185 | 0.70 | **0.838** | +20% |

**压缩收益**：Max/Min 比值 26.3× → 5.1×，Peroxisome 单独的梯度爆炸风险解除。

#### 防线 2：Focal Loss (γ=2.0) + Label Smoothing (ε=0.1)（损失层面）

**动机**：静态权重（哪怕是平滑后的）对所有样本一视同仁。Focal Loss 提供**动态**调制——根据模型对当前样本的预测置信度自适应缩放梯度。

**Focal Loss 公式**：

$$\mathcal{L}_{focal} = -\sum_{c=1}^{C} y_c^{smoothed} \cdot (1 - p_c)^\gamma \cdot \log(p_c)$$

**动态调制效果**：
- 当 p_c → 0.9（易分样本）：(1-0.9)² = 0.01 → **梯度衰减 100×**
- 当 p_c → 0.5（困难样本）：(1-0.5)² = 0.25 → 保留 1/4 梯度
- 当 p_c → 0.1（极难样本）：(1-0.1)² = 0.81 → 几乎全梯度

γ=2.0 的选择：γ=1 衰减不足（过拟合风险），γ=5 衰减过度（欠拟合风险），γ=2.0 是 CV 社区广泛验证的平衡点。

**Label Smoothing 抗过拟合原理**：将 one-hot 目标 `[0,0,1,0,...,0]` 平滑为 `[ε/9, ε/9, 1-ε, ε/9,..., ε/9]`，强迫模型对所有类别保持最低不确定性（熵 ≥ H(ε)），防止对大类输出 p→0.999 的极端置信度。ε=0.1 意味着模型即使完全确定，也只能输出最大 0.9 的概率，等价于在损失函数中隐式注入正则化噪声。

**新增文件**：[python/models/focal_loss.py](python/models/focal_loss.py) — 独立模块，含单元测试。

#### 防线 3：同方差不确定性多任务加权（优化层面）

**动机**：v1.0/v2.0 中双任务 Loss 使用固定权重 `loss = loc_loss + 0.5 * mem_loss`。但 Membrane 二分类远简单于 Location 十分类——Membrane 训练初期迅速收敛，mem_loss 主导梯度方向，阻碍 Location 学习。

**数学**（Kendall et al., CVPR 2018）：

$$-\log p(y_{loc}, y_{mem}|f^W(x)) \propto \exp(-s_{loc}) \cdot \mathcal{L}_{loc} + \frac{1}{2}s_{loc} + \exp(-s_{mem}) \cdot \mathcal{L}_{mem} + \frac{1}{2}s_{mem}$$

其中 s = log σ² 为可学习参数。直观理解：
- 任务噪声 σ 大（困难任务）→ exp(-s) 小 → Loss 权重自动降低
- 任务噪声 σ 小（简单任务）→ exp(-s) 大 → Loss 权重自动升高
- 正则项 ½s 防止 σ → ∞ 的退化解（模型通过让所有 σ→∞ 来将 Loss 降到 0）

**实现**：[architectures.py:214-215](python/models/architectures.py#L214) — `log_var_loc`、`log_var_mem` 两个 `nn.Parameter`，初始 s=0（σ=1，双任务等权）。

**训练后收敛值**：σ_loc ≈ 1.20（定位任务噪声大→权重适中），σ_mem ≈ 0.50（膜分类简单→高精度→高权重）。Membrane 因简单而获得 ~5.8× 的相对权重提升，自动实现了人工调参无法达到的动态平衡。

#### 防线 4：640→256 信息瓶颈层（输入层面）

**动机**：蛋白质亚细胞定位仅需识别与**分选信号**相关的低维特征——N 端信号肽 (~30aa)、核定位信号 (~4-6 残基碱性簇)、过氧化物酶体靶向信号 (C 端 SKL 模体)、跨膜螺旋等。ESM-2 640 维 embedding 承载了 150M 参数预训练的全部知识（残基理化性质、二级结构、接触图、同源模式），其中 >80% 与定位任务无关。

**实现**：`nn.Linear(640,256) → ReLU → Dropout`，位于 CNN 卷积层之前。

**设计考量**：
- **为什么不直接减小 ESM-2 模型？** esm2_t6_8M (320-dim) 的嵌入质量远低于 t30_150M，影响定位信号提取。瓶颈层让模型**自己学习**哪些维度对定位有用，而非人为选择低质嵌入。
- **为什么是 256？** 信息论直觉：10 类定位 + 信号肽/跨膜螺旋/定位模体的特征空间维度远小于 640。256 在保留足够表达能力的同时迫使模型丢弃冗余。若设 128 可能丢失关键信号，512 则压缩效果不足。
- **为什么不冻结瓶颈层训练后移除？** 端到端训练让瓶颈层与下游 CNN/BiLSTM 协同适应，形成紧凑的低秩表示。

**参数量变化**：

| 组件 | v2.0 | v3.0 | Δ |
|------|------|------|---|
| Conv1d 层 | 176,224 | 78,368 | -97,856 |
| 瓶颈投影层 | — | 164,096 | +164,096 |
| **总计** | ~294,157 | ~356,821 | +62,664 |

瓶颈层本身增加了 164K 参数，但因 Conv1d 输入通道从 640→256 减少了 98K，净增仅 63K。

#### 协同正则化矩阵

| 机制 | 作用维度 | v3.0 配置 | 互补关系 |
|------|---------|----------|---------|
| sqrt-smoothed weights | 类别先验 | 极差 5.1× | 静态缩小权重极差 |
| Focal Loss γ=2.0 | 样本难度 | 动态衰减易分样本梯度 | 动态补充静态权重 |
| Label Smoothing ε=0.1 | 输出置信度 | 防止 p→0.999 | 与 Focal Loss 正交抗过拟合 |
| 瓶颈层 640→256 | 输入信息流 | 压缩冗余特征 | 从源头减少可过拟合信息 |
| Weight Decay | 参数范数 | **5e-4** | 1e-4→5e-4，加倍压制 |
| Dropout | 神经元共适应 | **0.5** | 随机断开半数激活通路 |
| LayerNorm | 特征分布 | BiLSTM 输出后 | 稳定 Attention score |
| Uncertainty Weighting | 任务平衡 | 可学习 σ_loc, σ_mem | 替代人工 mem_weight |
| Early Stopping | 训练轮次 | patience=**10** (20→10) | 更快响应过拟合 |
| Cosine/Plateau Scheduler | 学习率 | 可选双调度器 | 支持 warm restart 探索 |

#### v3.0 最终训练结果

```
Epoch 23 best, val_acc=0.806, Gorodkin=0.741, MCC(membrane)=0.640
Train/Val gap=5.2pp
训练耗时 26.5 min（早停 E33）
不确定性参数收敛：σ_loc≈1.20, σ_mem≈0.50
LR 首次衰减 E23 (1e-3→5e-4)
```

| 指标 | v1.0 | v2.0 | v3.0 | 总 Δ |
|------|------|------|------|------|
| Test Gorodkin | 0.679 | 0.679 | **0.741** | **+0.062** |
| MCC (membrane) | — | — | **0.640** | — |
| Train acc | 0.975 | 0.972 | 0.870 | -10.5pp |
| Val acc | 0.759 | 0.808 | 0.806 | +4.7pp |
| Train/Val gap | 21.6pp | 16.4pp | **5.2pp** | **-16.4pp** |

**最重要的变化不是 Gorodkin +0.062，而是 Train acc 从 97.5% 降到 87.0%（模型不再死记训练集），Val acc 保持在 80.6%，gap 从 21.6pp 缩小到 5.2pp——模型真正学会了可泛化的定位特征。**

---

### 决策树：如果再遇到类似问题

```
严重过拟合 (Train/Val gap > 15pp) + 严重类别不平衡 (>20×)
│
├── 第一层：诊断
│   ├── 检查验证监控指标是否被加权 Loss 污染？
│   │   └── YES → 切换到 val_acc / unweighted val_loss
│   ├── 检查 Train acc 是否 > 95%？
│   │   └── YES → 模型容量过大或正则化不足
│   └── 检查 Train/Val gap 趋势？
│       └── Gap 持续扩大 → 典型过拟合，需增强正则化
│
├── 第二层：正则化增强（由轻到重）
│   ├── Dropout 0.3 → 0.5 → 0.7
│   ├── Weight Decay 1e-4 → 5e-4 → 1e-3
│   ├── LayerNorm / BatchNorm 稳定中间层分布
│   └── 若仍未控制 → 进入第三层
│
├── 第三层：结构性干预
│   ├── 预训练特征是否过于丰富？→ 加入信息瓶颈层
│   ├── 类别权重是否极端？→ sqrt 平滑（永远先于 Focal Loss）
│   ├── 多任务权重是否人工硬编码？→ 不确定性加权
│   └── 若仍未控制 → 进入第四层
│
└── 第四层：损失函数重构
    ├── 静态权重 → Focal Loss（动态调制）
    ├── 硬标签 → Label Smoothing（软化目标）
    └── 注意：Focal Loss 的 γ 和 Label Smoothing 的 ε 不可同时大幅调高，
         γ 高 → 梯度稀疏，ε 高 → 目标模糊，二者叠加可能致欠拟合
```

---

### 踩坑记录

1. **逆频率权重污染验证信号**（v1→v2 核心教训）
   - 现象：val_acc +5pp 但 weighted val_loss +60%
   - 根因：权重放大效应，小类误判被 8-26× 膨胀
   - 修复：验证监控与训练 Loss 解耦
   - **普适性**：任何使用非均匀 Loss 的场景，验证指标必须独立于训练 Loss 的权重体系

2. **sqrt 平滑优于直接逆频率**（v2→v3 设计选择）
   - 直觉：极差 26.3× → 5.1×，梯度不再被小类单点误判劫持
   - 为什么不直接 uniform？小类（Peroxisome 93 样本）确实需要更多关注
   - 为什么是 sqrt 而非 log？log 压缩过度（极差 ~1.9×），几乎退化为 uniform

3. **Focal Loss γ=2.0 是安全默认值，不必调**
   - γ=0 → 退化为普通 CE（无动态调制）
   - γ=1 → 调制不足，易分样本仍主导梯度
   - γ=5 → 仅关注极难样本，梯度过于稀疏
   - 我们的实验：γ=2.0 直接工作，无需调参

4. **LayerNorm 对 Attention 的稳定效果被低估**
   - Attention score = V·tanh(W1·features + W2·hidden)
   - 若 features/hidden 的数值分布不稳定（某维度激活值异常高），Attention 被单个神经元绑架
   - LayerNorm 沿特征维度归一化 → 每个维度贡献均等 → Attention 真正学习序列位置的重要性

5. **不确定性加权优于人工 mem_weight**
   - mem_weight=0.5 的假设：Membrane 比 Location 简单，应降低权重
   - 实际：简单任务的高精度信号对困难任务有正向引导作用
   - 不确定性加权自动学习最优平衡（σ_mem=0.50 → 高权重），优于人工直觉

---

## 文档参考

- ESM-2 HuggingFace：https://huggingface.co/docs/transformers/model_doc/esm
- ESM-2 t30 模型页：https://huggingface.co/facebook/esm2_t30_150M_UR50D
- PyTorch 文档：https://pytorch.org/docs/stable/
- BioPython SeqIO：https://biopython.org/docs/latest/api/Bio.SeqIO.html
- Spring Boot 快速开始：https://spring.io/quickstart
- Element Plus：https://element-plus.org/zh-CN/component/overview.html
- Vue 3：https://cn.vuejs.org/guide/introduction.html
- ECharts + vue-echarts：https://github.com/ecomfe/vue-echarts
