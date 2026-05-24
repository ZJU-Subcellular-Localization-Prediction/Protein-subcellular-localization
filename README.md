# 蛋白质亚细胞定位预测 — Web 应用

> 基于深度学习 CNN + BiLSTM + Bahdanau Attention 的蛋白质亚细胞定位预测系统，全栈 Web 应用。
>
> 原项目 [Protein-subcellular-localization](https://github.com/ClaudiaRaffaelli/Protein-subcellular-localization) 使用 TensorFlow/Keras，本项目以 **PyTorch + ESM-2 + Spring Boot + Vue 3** 全栈重构。

---

## 核心成果

| 指标 | 数值 | 说明 |
|------|------|------|
| **Gorodkin (Location MCC)** | **0.741** | 10 类亚细胞定位多类 MCC，超过原项目基线 |
| **MCC (Membrane)** | **0.640** | 膜结合状态二分类 |
| **Train/Val Gap** | **5.2pp** | Train 87.0% vs Val 80.6%，过拟合受控 |
| **推理耗时** | **~357ms** | ESM-2 编码 + 模型推理（GPU） |
| **最佳 Epoch** | **23** | Early Stopping patience=10，训练 33 epoch 早停 |

> 完整调优路线图（v1.0 → v2.0 → v3.0）见 [CLAUDE.md](./CLAUDE.md) 末尾【调优纪实档案】。

---

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 特征提取 | HuggingFace ESM-2 (`esm2_t30_150M_UR50D`) | 150M 参数蛋白质语言模型，640 维嵌入 |
| 深度学习 | PyTorch 2.x + CUDA 12.1 | CNN + BiLSTM + Bahdanau Attention |
| 后端 | Java 23 / Spring Boot 3.x + MyBatis-Plus | REST API，ProcessBuilder 调用 Python 推理 |
| 数据库 | MySQL 8.x | 序列 + 预测记录持久化 |
| 前端 | Vue 3 + Element Plus + ECharts 5 | SPA，SVG 细胞图 + 概率柱状图 + Attention 热力图 |
| 环境 | Anaconda (Python 3.10) | conda env `protein-local` |

---

## 快速开始

### 前置条件

- [Anaconda](https://docs.conda.io/en/latest/miniconda.html) 或 Miniconda
- [Node.js](https://nodejs.org/) ≥ 18
- [MySQL](https://dev.mysql.com/downloads/) 8.x 已启动
- [JDK](https://adoptium.net/) ≥ 17（编译后端）
- NVIDIA GPU + CUDA 12.1（训练/推理推荐，CPU 也可运行但较慢）

### 1. 克隆仓库 & 导入环境

```bash
git clone git@github.com:ZJU-Subcellular-Localization-Prediction/Protein-subcellular-localization.git
cd Protein-subcellular-localization

# 导入 conda 算法环境
conda env create -f python/environment.yml
conda activate protein-local
```

### 2. 下载 ESM-2 模型（首次，需联网 ~600MB）

```bash
python -c "from transformers import AutoTokenizer, AutoModel; \
  AutoTokenizer.from_pretrained('facebook/esm2_t30_150M_UR50D'); \
  AutoModel.from_pretrained('facebook/esm2_t30_150M_UR50D')"
```

### 3. 特征提取（如需重新训练模型）

```bash
cd python
python extract_features.py
# 输出：data/features/{train,val,test}/*.pt (13858 个文件) + manifest.pt
# 耗时约 1-2 小时（GPU）
```

> 如果仅运行推理（不重新训练），可跳过此步。仓库已包含预训练模型权重和 `manifest.pt`。

### 4. 模型训练（可选，已有预训练权重）

```bash
python train.py --model CNN_BLSTM_Attention --epochs 60 --batch_size 32 \
    --lr 0.001 --n_hid 64 --n_filt 32 --drop_prob 0.5 \
    --use_class_weights --use_uncertainty --monitor val_acc \
    --seed 42 --output python/outputs/cnn_blstm_attn_v3
```

### 5. 启动后端

```bash
# 创建数据库
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS protein_localization;"
mysql -u root -p protein_localization < backend/sql/init.sql

# 编辑 backend/src/main/resources/application.properties：
#   spring.datasource.password=你的MySQL密码
#   predict.python.path=C:/Users/.../miniconda3/envs/protein-local/python.exe

# 编译并启动（端口 8080）
cd backend
mvn compile spring-boot:run
```

### 6. 启动前端

```bash
cd frontend
npm install
npm run dev
# 打开浏览器 http://localhost:5173
```

---

## 核心算法架构（v3.0）

模型 `CNN_BLSTM_Attention` 采用**四重防线**对抗过拟合与类别不平衡：

### 防线 1：平方根平滑类别权重

原始逆频率权重极差 26.3×（Peroxisome ×8.94 vs Nucleus ×0.34），改用平方根平滑将极差压缩至 5.1×：

$$w_i = \sqrt{\frac{N_{total}}{C \cdot N_i}}$$

### 防线 2：Focal Loss (γ=2.0) + Label Smoothing (ε=0.1)

动态调制样本梯度——易分样本 (p≈0.9) 梯度衰减 100×，困难样本 (p≈0.1) 保留全梯度。Label Smoothing 将 one-hot 目标平滑为软标签，防止模型输出极端置信度。

$$\mathcal{L}_{focal} = -\sum_{c=1}^{C} y_c^{smoothed} \cdot (1 - p_c)^\gamma \cdot \log(p_c)$$

### 防线 3：同方差不确定性多任务加权

可学习参数 σ_loc、σ_mem 自动平衡双任务 Loss，替代人工调参。训练收敛时 σ_loc≈1.20（定位任务噪声大→权重适中）、σ_mem≈0.50（膜分类简单→高精度→高权重）。

$$\mathcal{L} = \exp(-s_{loc})\mathcal{L}_{loc} + \tfrac{1}{2}s_{loc} + \exp(-s_{mem})\mathcal{L}_{mem} + \tfrac{1}{2}s_{mem}$$

### 防线 4：640→256 信息瓶颈层

ESM-2 640 维嵌入含大量与定位无关的语义信息（二级结构、接触图、同源模式），`Linear(640,256)→ReLU→Dropout` 强制压缩至定位相关低维子空间，同时减少卷积层参数 98K。

### 协同正则化矩阵

| 机制 | 作用维度 | 配置 |
|------|---------|------|
| Weight Decay | 参数范数 | 5e-4 |
| Dropout | 神经元共适应 | 0.5 |
| LayerNorm | 特征分布 | BiLSTM 输出后，稳定 Attention score |
| Early Stopping | 训练轮次 | patience=10，监控 val_acc |

### 数据流

```
用户输入 (AA 序列)
  → 前端实时校验（非法字符检测）
  → POST /api/predict
  → Java PredictService.cleanSequence()
  → ProcessBuilder("python predict.py --sequence ...")
  → ESM-2 (序列 → 640-dim embedding)
  → 640→256 瓶颈层
  → 2×Conv1d(k=3,5) → Concat → Conv1d(k=3)
  → BiLSTM(64, bidirectional) → LayerNorm
  → Bahdanau Attention → Dense(128) → Dropout
  → 双头输出: Location(10) + Membrane(2)
  → JSON 响应 → Vue 3 渲染（细胞图 + 概率图 + 热力图）
```

---

## 数据集概况

### 数据来源

**DeepLoc** (Almagro Armenteros et al., *Bioinformatics*, 2017)，存储于 RTF 封装的 FASTA 格式。每条序列头承载 `>UniProtID Location-Membrane [test]` 三元组元数据。

### 划分机制

| Split | 样本数 | 占比 | 规则 |
|-------|--------|------|------|
| Train | 8,313 | 60.0% | 4-fold 循环，fold ∈ {2,3,4} |
| Val | 2,772 | 20.0% | fold = 1 |
| Test | 2,773 | 20.0% | FASTA 头 `test` 标记 |
| **总计** | **13,858** | 100% | — |

### 任务一：亚细胞定位（10 分类，26 倍不平衡）

| 类别 | Train | 占比 |
|------|-------|------|
| Nucleus | 2,427 | **29.1%** |
| Cytoplasm | 1,525 | 18.3% |
| Extracellular | 1,185 | 14.2% |
| Mitochondrion | 906 | 10.9% |
| Cell membrane | 800 | 9.7% |
| ER | 517 | 6.2% |
| Plastid | 453 | 5.4% |
| Golgi apparatus | 215 | 2.6% |
| Lysosome/Vacuole | 192 | 2.3% |
| **Peroxisome** | **93** | **1.1%** |

### 任务二：膜结合状态（实际二分类，M:S≈1.7:1）

原始定义含 `U`(Unknown) 但全量数据中无 U 样本，M:S≈1.7:1，中度不平衡。

### 序列预处理

- **中心截断**（>1000 AA）：从序列中心切除多余氨基酸，保护 N 端信号肽和 C 端定位信号
- **末端零填充**（<1000 AA）：DataLoader 阶段动态补零至 (1000, 640)
- **特征存储**：每条序列存为独立 float16 `.pt` 文件（~625 KB/条），训练时按需加载，避免全量 ~35GB 内存占用

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/predict` | 提交蛋白质序列，返回预测结果 |
| GET | `/api/history?page=1&size=20` | 查询历史记录（分页） |
| GET | `/api/history/{id}` | 查询单条记录详情 |

### POST /api/predict

请求体：
```json
{"sequence": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHL..."}
```

成功响应 (200)：
```json
{
  "code": 200,
  "data": {
    "sequence_id": "a1b2c3d4e5f6",
    "predicted_location": "Extracellular",
    "location_confidence": 0.741,
    "predicted_membrane": "Soluble",
    "membrane_confidence": 0.825,
    "all_probabilities": { "Cell membrane": 0.016, "Cytoplasm": 0.068, ... },
    "attention_weights": [[0.001], [0.003], ...],
    "model_version": "v3",
    "inference_time_ms": 357
  }
}
```

### 错误响应码

| 状态码 | 含义 | 典型场景 |
|--------|------|---------|
| 400 | 请求参数错误 | 空序列 / 含非氨基酸字符 |
| 502 | Python 推理失败 | GPU 显存不足 / 模型文件损坏 |
| 503 | 推理服务不可用 | Python 脚本缺失 / 环境配置错误 |
| 504 | 推理超时 | 推理超过 60s（ProcessBuilder 强制终止） |
| 500 | 服务器内部错误 | 数据库连接失败 / 未知异常 |

---

## 全栈鲁棒性

### 超时保护链

```
前端 (axios, 65s timeout)
  → 后端 (ProcessBuilder.waitFor, 60s timeout)
    → Python (~357ms 正常)
```

超时未完成 → `destroyForcibly()` → 504 Gateway Timeout。

### 数据清洗流水线

```
用户输入 → 前端实时校验（非法字符警告）
  → Java PredictService.cleanSequence(): 转大写 → 去空白 → 仅保留 20 种标准 AA
  → Python predict.py: 二次 clean + pad_center_truncate(1000)
```

### 前端降级策略

| 场景 | 处理方式 |
|------|---------|
| 后端不可达 | axios interceptor → `ElMessage.error` |
| 超长序列 (>1000aa) | 前端 info 提示"将中心截断" |
| 非法字符 | 实时黄色警告 + 后端自动清洗 |
| Attention 数据缺失 | 热力图区域完全隐藏 |
| 概率数据全零/空 | SVG 细胞图半透明 + 占位提示 |

### 离线环境支持

后端自动设置 `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`，ESM-2 模型从本地 HuggingFace 缓存加载，无需联网。

---

## 模型架构（7 种对比实验）

| 序号 | 架构 | 说明 |
|------|------|------|
| 1 | FFN | Feed-Forward Network（baseline） |
| 2 | CNN | 双分支 Conv1d(k=3,5) → Concat → Conv → MaxPool |
| 3 | BLSTM | 双向 LSTM |
| 4 | CNN+BLSTM | CNN 特征提取 + 双向 LSTM |
| 5 | BLSTM+Attention | 双向 LSTM + Bahdanau Attention |
| 6 | **CNN+BLSTM+Attention** | **CNN + BiLSTM + Attention（v3.0 最佳模型）** |
| 7 | CNN+BLSTM+Attention_complete | 6 并行 Conv1d(k=1,3,5,9,15,21) → BiLSTM → Attention |

---

## 评估指标

| 指标 | 说明 |
|------|------|
| **Gorodkin** | 多类 Matthews 相关系数（MCC），评估 10 类定位分类质量 |
| **MCC** | 二元 Matthews 相关系数，评估膜结合状态分类 |
| **混淆矩阵** | 10×10（定位）+ 3×3（膜结合），含热力图可视化 |
| **Attention 权重** | ECharts 热力图，展示模型在序列上的关注区域 |

---

## 开发进度

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 0 | 项目初始化 | ✅ |
| Phase 1 | ESM-2 特征提取（13,858 条序列） | ✅ |
| Phase 2 | PyTorch 模型重写（7 种架构 + 训练/评估） | ✅ |
| Phase 2.6 | 训练优化（加权 Loss + L2 + LR 调度） | ✅ |
| Phase 2.7 | P0+P1 验证解耦 + 正则化增强 | ✅ |
| Phase 2.8 | v3.0 多重防御重构 | ✅ |
| Phase 3 | Java 后端 + MySQL | ✅ |
| Phase 4 | Vue 3 前端（6 组件 + 4 页面） | ✅ |
| Phase 5 | 联调测试（E2E + 边界处理 + Code Review） | ✅ |

---

## 目录结构

```
Protein-subcellular-localization/
├── CLAUDE.md                    # 项目开发计划 + 数据契约 + 调优档案
├── README.md                    # 本文件
├── python/
│   ├── environment.yml          # conda 环境文件
│   ├── requirements.txt         # pip 依赖
│   ├── extract_features.py      # ESM-2 特征提取
│   ├── train.py                 # 模型训练（含 v3.0 四重防线）
│   ├── predict.py               # 推理脚本（供 Java 后端调用）
│   ├── models/
│   │   ├── attention.py         # Bahdanau Attention
│   │   ├── architectures.py     # 7 种架构（含 Complete）
│   │   └── focal_loss.py        # Focal Loss + Label Smoothing
│   ├── data/
│   │   ├── dataset.py           # PyTorch Dataset + DataLoader
│   │   └── features/            # ESM-2 特征 (.pt 文件 + manifest.pt)
│   └── outputs/
│       ├── cnn_blstm_attn_v2/   # v2.0 训练产物
│       └── cnn_blstm_attn_v3/   # v3.0 训练产物（当前部署模型）
├── backend/
│   ├── src/main/java/com/proteinlocal/
│   │   ├── config/              # MyBatis-Plus 配置
│   │   ├── controller/          # REST Controller + 全局异常处理
│   │   ├── dto/                 # 请求/响应 DTO
│   │   ├── entity/              # 数据库实体
│   │   ├── exception/           # PredictException（精确错误码）
│   │   ├── mapper/              # MyBatis-Plus Mapper
│   │   └── service/             # PredictService（ProcessBuilder 调用 Python）
│   ├── src/main/resources/application.properties
│   ├── pom.xml
│   └── sql/init.sql
├── frontend/
│   ├── src/
│   │   ├── views/               # Home, Predict, History, About
│   │   ├── components/          # SequenceInput, ResultCard, CellDiagram,
│   │   │                        # ProbabilityChart, AttentionHeatmap, HistoryTable
│   │   ├── router/index.js
│   │   ├── api/                 # Axios 封装 + 拦截器
│   │   └── App.vue
│   ├── package.json
│   └── vite.config.js
└── docs/
    └── 小组分工计划书.md
```

---

## 参考

- 原项目：https://github.com/ClaudiaRaffaelli/Protein-subcellular-localization (MIT)
- 核心论文：Almagro Armenteros JJ, et al. *DeepLoc: prediction of protein subcellular localization using deep learning*. Bioinformatics, 2017.
- ESM-2：https://huggingface.co/docs/transformers/model_doc/esm
- Focal Loss：Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017
- 不确定性加权：Kendall et al., *Multi-Task Learning Using Uncertainty*, CVPR 2018

---

## 团队

CS 方向 + 医学方向跨学科团队
