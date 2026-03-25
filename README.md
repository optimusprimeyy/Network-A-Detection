# Network-A-Detection
**基于 GBAE + LightGBM 多算法融合的网络流量异常检测系统**
## 项目亮点
- **解决真实业务三大痛点**
  1. 样本不平衡（攻击样本稀少）
  2. 冷启动 / 无标签数据检测
  3. 已知攻击 + 未知攻击无法兼顾

- **双层模型融合架构**
  - **LightGBM**：监督学习，高精度识别已知攻击
  - **GBAE（粒球自编码器）**：无监督学习，专用于冷启动与未知威胁
  - **加权融合**：兼顾精度与泛化，效果优于单模型

- **可复现、轻量、易部署**
  适用于：网络入侵检测、风控异常识别、工业异常检测、安全运营。

---

## 实验结果
| 模型 | AUC | 能力说明 |
|------|-----|----------|
| GBAE (无监督) | 0.7860 | 冷启动 / 未知攻击 |
| LightGBM (监督) | 0.9547 | 已知攻击高精度 |
| **融合模型** | **0.9413** | **兼顾已知 + 未知攻击** |

---

## 项目结构
```
Network-A-Detection/
├── UAD/                 # GBAE 无监督异常检测
├── Utils/               # 数据预处理 & 工具函数
├── models/              # 训练好的模型文件
├── logs/                # 训练日志
├── cold_start.py        # 冷启动检测
├── fusion_model.py      # 多模型融合推理
├── train_gbae.py        # 训练 GBAE
├── train_lgb.py         # 训练 LightGBM
└── README.md
```

---

## 使用方法
```bash
# 训练 LightGBM（二分类 + 多分类）
python train_lgb.py

# 训练 GBAE 冷启动模型
python train_gbae.py

# 模型融合推理
python fusion_model.py
```

---

## 适用场景
- Network Intrusion Detection
- Anomaly Detection for Security
- Risk Control / Fraud Detection
- Cold-start & Unknown Threat Detection

---

## 求职方向
安全算法、风控算法、异常检测、机器学习算法岗
