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
│   ├── src/
│   ├── pom.xml
│   └── sql/
│       └── init.sql             # 数据库建表 DDL
├── frontend/                    # Vue 3 前端
│   ├── src/
│   │   ├── views/               # 页面组件
│   │   ├── components/          # 通用组件
│   │   ├── router/              # 路由配置
│   │   └── api/                 # API 请求封装
│   ├── package.json
│   └── vite.config.js
└── docs/                        # 参考文档
    └── 小组分工计划书.md
```

---

## 快速开始

### 1. 环境准备

```bash
# 创建 Anaconda 环境
conda create -n protein-local python=3.10 -y
conda activate protein-local

# 安装 PyTorch（GPU 版本）
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 安装其他依赖
pip install transformers biopython scikit-learn matplotlib pandas
```

### 2. 特征提取

```bash
cd python
python extract_features.py
# 生成 dataset_esm2_1000.pt
```

### 3. 模型训练

```bash
python train.py
# 生成 best_model.pt
```

### 4. 启动后端（Spring Boot）

```bash
cd backend
mvn spring-boot:run
# 启动在 http://localhost:8080
```

### 5. 启动前端（Vue 3）

```bash
cd frontend
npm install
npm run dev
# 启动在 http://localhost:5173
```

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/predict` | 提交蛋白质序列，返回预测结果 |
| GET | `/api/history` | 查询历史预测记录（分页） |
| GET | `/api/history/{id}` | 查询单条记录详情 |

详细接口规范见 [小组分工计划书附录 B](docs/小组分工计划书.md)。

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
