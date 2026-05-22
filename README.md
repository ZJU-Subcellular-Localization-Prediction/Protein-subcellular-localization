# 蛋白质亚细胞定位预测 — Web 应用

> 基于 [Protein-subcellular-localization](https://github.com/ClaudiaRaffaelli/Protein-subcellular-localization) 开源项目进行技术栈现代化重构。
>
> 原项目使用 TensorFlow/Keras，本项目改用 **PyTorch + ESM-2 + Java + Vue 3** 全栈方案。

---

## 项目简介

蛋白质亚细胞定位是生物信息学中的多分类问题——给定一条蛋白质氨基酸序列，预测该蛋白质在细胞中的位置（如细胞核、线粒体、细胞膜等 10 种类别），同时预测其膜结合状态（膜蛋白 / 可溶性蛋白）。

本项目使用深度学习中的 **CNN + 双向 LSTM + Bahdanau 注意力机制** 进行预测，并构建了完整的前后端 Web 应用进行系统演示。

---

## 技术栈

| 层级 | 技术 | 版本 |
|------|------|------|
| 特征提取 | HuggingFace ESM-2 (`esm2_t30_150M_UR50D`) | 640 维 |
| 深度学习框架 | PyTorch + CUDA | 2.x |
| 后端 | Java / Spring Boot | 3.x |
| 数据库 | MySQL | 8.x |
| 前端 | Vue 3 + Element Plus + ECharts | 3.x |
| 环境管理 | Anaconda | — |

---

## 目录结构

```
Protein-subcellular-localization/
├── CLAUDE.md                    # 项目进度缓存 + 开发备忘
├── README.md                    # 本文件
├── python/                      # Python 模块
│   ├── environment.yml          # conda 环境文件
│   ├── requirements.txt         # pip 依赖
│   ├── extract_features.py      # ESM-2 特征提取脚本
│   ├── train.py                 # 模型训练脚本
│   ├── predict.py               # 推理脚本（供 Java 后端调用）
│   ├── models/                  # PyTorch 模型定义
│   │   ├── attention.py         # Bahdanau Attention 层
│   │   ├── architectures.py     # 全部 7 种对比架构
│   │   └── complete.py          # 最终完整版模型
│   └── data/
│       └── dataset.py           # PyTorch Dataset + DataLoader
├── backend/                     # Java Spring Boot 后端
│   ├── src/main/java/com/proteinlocal/
│   │   ├── config/              # MyBatis-Plus 配置
│   │   ├── controller/          # REST Controller + 全局异常处理
│   │   ├── dto/                 # 请求/响应 DTO（含 @JsonProperty 映射）
│   │   ├── entity/              # 数据库实体
│   │   ├── exception/           # PredictException（精确错误码）
│   │   ├── mapper/              # MyBatis-Plus Mapper
│   │   └── service/             # 业务逻辑（PredictService + ProcessBuilder）
│   ├── src/main/resources/application.properties
│   ├── pom.xml
│   └── sql/init.sql
├── frontend/                    # Vue 3 + Element Plus 前端
│   ├── src/
│   │   ├── views/               # Home, Predict, History, About
│   │   │   ├── Home.vue
│   │   │   ├── Predict.vue      # 核心预测页（状态宿主 + API 调用）
│   │   │   ├── History.vue
│   │   │   └── About.vue
│   │   ├── components/          # 6 个通用子组件
│   │   │   ├── SequenceInput.vue    # 序列输入 + FASTA 上传 + 实时校验
│   │   │   ├── ResultCard.vue       # 预测结果卡片
│   │   │   ├── CellDiagram.vue      # SVG 手绘细胞结构图
│   │   │   ├── ProbabilityChart.vue # ECharts 概率柱状图
│   │   │   ├── AttentionHeatmap.vue # Attention 权重热力图
│   │   │   └── HistoryTable.vue     # 历史记录表格
│   │   ├── router/index.js      # Vue Router（懒加载）
│   │   ├── api/                 # Axios 封装 + ApiResponse 解包拦截器
│   │   └── App.vue              # 全局布局（Header + Menu + Footer）
│   ├── package.json
│   └── vite.config.js           # @ 别名 + /api 代理
└── docs/                        # 参考文档
    └── 小组分工计划书.md
```

---

## 开发进度

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 0 | 项目初始化 | ✅ 完成 |
| Phase 1 | ESM-2 特征提取（13858 条序列，文件级存储） | ✅ 完成 |
| Phase 2 | PyTorch 模型重写（7 种架构 + 训练/评估脚本） | ✅ 完成 |
| Phase 3 | Java 后端 + MySQL（Spring Boot + MyBatis-Plus） | ✅ 完成 |
| Phase 4 | Vue 3 前端（6 组件 + 4 页面 + ECharts + SVG） | ✅ 完成 |
| Phase 5 | 联调测试（E2E 全链路 + 边界处理 + Code Review） | ✅ 完成 |

> **剩余工作**：模型正式训练（60 epoch 超参数搜索）、Gorodkin 评估指标验证。

---

## 快速开始

### 1. 环境准备

```bash
# 创建 Anaconda 环境
conda create -n protein-local python=3.10 -y
conda activate protein-local

# 安装 PyTorch（GPU 版本，CUDA 12.1）
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 安装其他 Python 依赖
pip install transformers biopython scikit-learn matplotlib pandas

# 首次运行需下载 ESM-2 模型（~600 MB），确保网络通畅
# 之后可设置离线模式跳过网络检查（见下文配置）
```

### 2. 特征提取

```bash
cd python
python extract_features.py
# 输出：data/features/{train,val,test}/XXXXX.pt (13858 个独立文件) + manifest.pt
# 耗时约 1-2 小时（GPU），取决于序列数量
```

### 3. 模型训练

```bash
python train.py --model CNN_BLSTM_Attention --epochs 60 --batch_size 64 \
                --lr 0.001 --n_hid 64 --n_filt 32 --drop_prob 0.3 --drop_hid 0.3
# 输出：outputs/{model}/best_model.pt + training_curves.png
```

### 4. 配置并启动后端

**前置条件**：MySQL 8.x 已安装并运行。

```bash
# 1. 创建数据库和表
mysql -u root -p < backend/sql/init.sql

# 2. 修改 backend/src/main/resources/application.properties：
#    - spring.datasource.password=你的MySQL密码
#    - predict.python.path=C:/Users/.../miniconda3/envs/protein-local/python.exe
#    - predict.model.path=../outputs/cnn_blstm_attn/best_model.pt

# 3. 编译并启动
cd backend
mvn compile spring-boot:run
# 启动在 http://localhost:8080
```

> **中国区用户注意**：后端已默认配置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1` 环境变量，调用 Python 时会跳过 HuggingFace Hub 连接。首次使用前请确保 ESM-2 模型已下载到本地缓存。

### 5. 启动前端（Vue 3）

```bash
cd frontend
npm install
npm run dev
# 启动在 http://localhost:5173
# Vite 已配置 proxy：/api → http://localhost:8080
```

---

## 团队协作快速启动（零门槛开箱指南）

> 以下指南面向从零开始克隆仓库的队友，按顺序执行即可在本地跑通完整全栈服务。

### Step 1 — 克隆仓库

```bash
git clone git@github.com:ZJU-Subcellular-Localization-Prediction/Protein-subcellular-localization.git
cd Protein-subcellular-localization
```

### Step 2 — 导入 Conda 算法环境

项目维护者在首次配置好环境后，应导出环境文件供队友一键复现：

**维护者（环境导出）**：
```bash
conda activate protein-local
conda env export --no-builds > python/environment.yml
git add python/environment.yml
git commit -m "chore: export conda environment.yml"
git push
```

**队友（环境导入）**：
```bash
conda env create -f python/environment.yml
conda activate protein-local
```

### Step 3 — 下载 ESM-2 模型（首次，需联网）

```bash
# 模型约 600 MB，下载到本地 HuggingFace 缓存
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('facebook/esm2_t30_150M_UR50D'); AutoModel.from_pretrained('facebook/esm2_t30_150M_UR50D')"
```

> 下载完毕后，后端会自动以离线模式调用（`HF_HUB_OFFLINE=1`），无需再次联网。

### Step 4 — 生成特征文件（可选，耗时较长）

```bash
cd python
python extract_features.py
# 约 1-2 小时（GPU），生成 data/features/ 下 13858 个 .pt 文件（约 8 GB）
```

> 如果只需要测试推理功能（不重新训练模型），可跳过此步。仓库中已包含 `manifest.pt`（标签索引）和预训练模型权重。

### Step 5 — 配置并启动后端

```bash
# 1. 创建数据库
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS protein_localization;"
mysql -u root -p protein_localization < backend/sql/init.sql

# 2. 编辑 backend/src/main/resources/application.properties
#    修改 spring.datasource.password 和 predict.python.path

# 3. 启动
cd backend
mvn compile spring-boot:run
```

### Step 6 — 启动前端

```bash
cd frontend
npm install
npm run dev
# 打开浏览器 http://localhost:5173 → Predict → Load Example → Submit
```

### 预训练模型权重

| 文件 | 大小 | 获取方式 |
|------|------|---------|
| `outputs/*/best_model.pt` | ~3.5 MB | **Git 仓库已包含**（GitHub 可直接托管） |
| `python/data/features/train/` | ~5 GB | 运行 `extract_features.py` 生成 |
| `python/data/features/val/` | ~1.5 GB | 运行 `extract_features.py` 生成 |
| `python/data/features/test/` | ~1.5 GB | 运行 `extract_features.py` 生成 |
| `python/data/features/manifest.pt` | ~700 KB | **Git 仓库已包含** |

### 常见问题

| 问题 | 解决方案 |
|------|---------|
| `ModuleNotFoundError: No module named 'torch'` | 确认已激活 conda 环境：`conda activate protein-local` |
| `Cannot connect to backend` | 检查 Spring Boot 是否启动在 8080 端口 |
| Python 推理超时 / 500 错误 | 检查 `predict.python.path` 是否正确指向 conda env 的 `python.exe` |
| `HF_HUB_OFFLINE=1` 报错找不到模型 | 执行 Step 3 下载 ESM-2 模型到本地缓存 |
| MySQL `Access denied` | 检查 `application.properties` 中 `spring.datasource.password` |
| `npm: command not found` | 安装 Node.js ≥18：https://nodejs.org/ |

---

## API 接口

### 接口列表

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/predict` | 提交蛋白质序列，返回预测结果 |
| GET | `/api/history?page=1&size=20` | 查询历史预测记录（分页） |
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
    "model_version": "v1",
    "inference_time_ms": 357
  }
}
```

### 错误响应码

| 状态码 | 含义 | 典型场景 |
|--------|------|---------|
| 400 | 请求参数错误 | 空序列 / 含非氨基酸字符 |
| 502 | Python 推理失败 | GPU 显存不足 / 模型文件损坏 / JSON 解析失败 |
| 503 | 推理服务不可用 | Python 脚本缺失 / Python 环境配置错误 |
| 504 | 推理超时 | 推理超过 60s 上限（ProcessBuilder 强制终止） |
| 500 | 服务器内部错误 | 数据库连接失败 / 未知运行时异常 |

所有错误响应遵循统一格式：`{"code": <HTTP码>, "message": "<人类可读描述>", "data": null}`

---

## 全栈鲁棒性与异常处理

本节阐述系统在长耗时推理、网络异常、模型故障等极端边界情况下的优雅降级策略。

### 1. 推理超时保护

```
前端 (axios, 65s timeout)
  → 后端 (ProcessBuilder.waitFor, 60s timeout)
    → Python (GPU 推理, ~357ms 正常)
```

- 后端 60s 超时未完成 → `process.destroyForcibly()` 强制终止 → 返回 **504 Gateway Timeout**
- 前端 axios 65s 超时 → 提示"Request timeout — inference took longer than 65s"
- Loading 动画分 4 阶段推进（`Cleaning sequence → ESM-2 feature extraction → Model inference → Almost done`）

### 2. 异常分类与精确错误码

后端使用 `PredictException(statusCode, message)` 细分 5 类故障，`GlobalExceptionHandler` 映射为对应 HTTP 状态码：

| 异常来源 | HTTP 码 | 后端处理 |
|---------|--------|---------|
| 序列为空 / 非法字符 | 400 | `IllegalArgumentException` + `@Valid` 校验 |
| Python 非零退出 | 502 | 捕获 stderr 并嵌入错误消息 |
| JSON 解析失败 | 502 | `JsonProcessingException` 单独捕获 |
| 脚本/Python 不存在 | 503 | `ProcessBuilder.start()` IOException 包装 |
| 子进程超时 | 504 | `waitFor(60s)` 返回 false → `destroyForcibly()` |

### 3. 前端降级策略

| 场景 | 处理方式 |
|------|---------|
| 后端不可达 | axios interceptor → `ElMessage.error("Cannot connect to backend...")` |
| 预测成功 | `ElMessage.success("Prediction complete — Extracellular (74.1%) in 357ms")` |
| 预测失败 | 内联 `el-alert` (持久) + `ElMessage.error` (即时 toast) |
| 超长序列 (>1000aa) | 前端 `el-alert` info 警告"将中心截断至 1000aa" |
| 非法字符 | 实时校验 + 黄色 `el-alert` 警告 + 后端自动清洗 |
| Attention 数据缺失 (Complete 模型) | 热力图区域完全隐藏 |
| 概率数据全零/空 | CellDiagram SVG 半透明 + "No prediction data" 占位 |
| 多头 Attention | 自动均值聚合为单值热力图 |

### 4. 数据清洗流水线

```
用户输入 → 前端实时校验（非法字符警告）
  → POST /api/predict
    → PredictService.cleanSequence():
      1. null → ""
      2. 转大写
      3. 去空格/换行
      4. 去除非标准氨基酸字符（仅保留 ACDEFGHIKLMNPQRSTVWY）
    → isEmpty → 400
  → Python predict.py: 二次 clean + pad_center_truncate(1000)
```

### 5. 离线环境支持

针对中国区 HuggingFace Hub 访问受限问题，后端在 `ProcessBuilder` 启动时注入环境变量：

```
HF_HUB_OFFLINE=1        # 禁止连接 HuggingFace Hub
TRANSFORMERS_OFFLINE=1  # 禁止 transformers 在线检查
```

ESM-2 模型需预先下载到本地 HuggingFace 缓存目录（`~/.cache/huggingface/hub/`）或通过 `TRANSFORMERS_CACHE` 环境变量指定路径。

---

## 模型架构（7 种对比实验）

| 序号 | 架构 | 说明 |
|------|------|------|
| 1 | FFN | Feed-Forward Network（baseline） |
| 2 | CNN | 双分支 Conv1d(k=3,5) → Concat → Conv → MaxPool |
| 3 | BLSTM | 双向 LSTM |
| 4 | CNN+BLSTM | CNN 特征提取 + 双向 LSTM |
| 5 | BLSTM+Attention | 双向 LSTM + Bahdanau Attention |
| 6 | CNN+BLSTM+Attention | CNN + BiLSTM + Attention（最佳对比模型） |
| 7 | CNN+BLSTM+Attention_complete | **最终模型**：6 并行 Conv1d(k=1,3,5,9,15,21) → BiLSTM → Attention |

---

## 评估指标

- **Gorodkin 度量**：多类 Matthews 相关系数（MCC），评估 10 类亚细胞定位分类质量
- **MCC**：二元 Matthews 相关系数，评估膜结合状态分类质量
- **混淆矩阵**：10×10（亚细胞定位）+ 3×3（膜结合状态）
- **Attention 权重可视化**：展示模型在序列上的关注区域

---

## 参考原项目

- 原项目仓库：https://github.com/ClaudiaRaffaelli/Protein-subcellular-localization
- 原项目许可证：MIT
- 核心参考论文：Almagro Armenteros JJ, et al. *DeepLoc: prediction of protein subcellular localization using deep learning*. Bioinformatics, 2017.

---

## 团队

CS 方向 + 医学方向跨学科团队
