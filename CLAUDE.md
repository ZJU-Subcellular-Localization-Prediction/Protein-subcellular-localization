# CLAUDE.md — CS 同学代码实现计划 & 进度追踪

---

## 项目概况

- **项目名**：蛋白质亚细胞定位预测 Web 应用
- **当前目录**：`D:\test\Protein-subcellular-localization`
- **原项目**：`D:\test\Protein-subcellular-localization-main`
- **环境管理**：Anaconda（conda env）
- **截止日期**：2026 年 5 月 28 日，剩余 8 天
- **GPU 状态**：有 GPU（CUDA 12.1）
- **ESM-2 模型**：`facebook/esm2_t30_150M_UR50D`（640 维嵌入，150M 参数）

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

### 3 类膜结合状态（membrane）

```python
labels_dic_membrane = {'M': 0, 'S': 1, 'U': 2}
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
    model_version VARCHAR(32) DEFAULT 'v1',
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
    "model_version": "v1",
    "inference_time_ms": 1523
  }
}
```
	
	**错误响应码**：
	
	| HTTP 码 | 含义 | 触发条件 |
	|---------|------|---------|
	| 400 | 请求参数错误 | 空序列、非氨基酸字符、`@Valid` 校验失败 |
	| 502 | Python 推理失败 | 模型崩溃、GPU 显存不足、ESM-2 加载失败 |
	| 503 | 推理服务不可用 | Python 脚本未找到、Python 可执行文件不存在 |
	| 504 | 推理超时 | 超过 `predict.python.timeout-seconds`（默认 60s） |
	| 500 | 内部未知错误 | 未预期的运行时异常 |

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

---

## Vue 组件树 & 路由

```
App.vue
├── Home.vue          (/)              — 项目介绍 + 技术架构
├── Predict.vue       (/predict)       — 核心预测页
│   ├── SequenceInput.vue              — 序列输入（文本框 + 上传 + 示例）
│   ├── ResultCard.vue                 — 预测结果卡片
│   ├── CellDiagram.vue                — SVG 细胞结构图（动态高亮）
│   ├── ProbabilityChart.vue           — ECharts 概率分布柱状图
│   └── AttentionHeatmap.vue           — Attention 权重热力图
├── History.vue       (/history)       — 历史记录
│   └── HistoryTable.vue               — el-table + el-pagination
└── About.vue         (/about)         — 团队 + 技术栈
```

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

> ⚠️ **训练状态备注（2026-05-22）**：FFN 2 epoch 冒烟测试 loss 下降正常，但当前超参数下准确率较低（loc≈66%, mem≈76%）。正式训练需进行超参数调优（lr、n_filt、n_hid、drop_prob 网格搜索），参见下方超参数搜索空间。调优将在 Phase 5 联调完成后专门进行。

> **阻塞点**：完整版 6 并行 Conv 层参数量 + GPU 显存管理。建议先用 `esm2_t12_35M`（480 维）快速跑通，再换 `esm2_t30_150M`（640 维）做最终训练。

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

> **注意**：`predict.py` 每次调用都重新加载 ESM-2 + PyTorch 模型（~3-5s），Phase 5 可优化为常驻 Flask 微服务。当前模型未正式训练，predict.py 语法验证通过但需训练后才能端到端推理。

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
- [ ] `python train.py --model CNN_BLSTM_Attention --epochs 60` 完成训练
- [ ] Gorodkin 值 > 0.6（与原项目趋势一致）
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
│   │   └── complete.py          # 最终完整版模型
│   └── data/
│       ├── __init__.py
│       ├── dataset.py           # PyTorch Dataset（文件级按需加载 + 动态 padding）
│       └── features/            # Phase 1 产出：独立 .pt 文件
│           ├── train/00000.pt ...
│           ├── val/00000.pt ...
│           ├── test/00000.pt ...
│           └── manifest.pt      # 索引 + labels + 元信息
├── backend/
│   ├── src/main/java/...
│   ├── src/main/resources/application.properties
│   ├── pom.xml
│   └── sql/
│       └── init.sql             # 建表 DDL
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

## 文档参考

- ESM-2 HuggingFace：https://huggingface.co/docs/transformers/model_doc/esm
- ESM-2 t30 模型页：https://huggingface.co/facebook/esm2_t30_150M_UR50D
- PyTorch 文档：https://pytorch.org/docs/stable/
- BioPython SeqIO：https://biopython.org/docs/latest/api/Bio.SeqIO.html
- Spring Boot 快速开始：https://spring.io/quickstart
- Element Plus：https://element-plus.org/zh-CN/component/overview.html
- Vue 3：https://cn.vuejs.org/guide/introduction.html
- ECharts + vue-echarts：https://github.com/ecomfe/vue-echarts
