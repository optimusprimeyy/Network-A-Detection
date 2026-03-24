import logging

import pandas as pd
import numpy as np
import os
import joblib
from lightgbm import early_stopping
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score
import lightgbm as lgb
from Utils.data_process import load_kdd_data, preprocess_binary_label, preprocess_features, load_config
from sklearn.model_selection import train_test_split
os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/binary_train.log", encoding="utf-8"),  # 日志保存路径
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# 1. 加载配置
config = load_config()

# 2. 数据加载
train_df, test_df = load_kdd_data()

train_df = preprocess_binary_label(train_df)
test_df = preprocess_binary_label(test_df)

X_train, y_train, X_test, y_test = preprocess_features(train_df, test_df, config)

# 3. 模型训练

# 3.1 样本不平衡
pos = sum(y_train == 1)
neg = sum(y_train == 0)
pos_weight = neg / pos

# 3.2
params = config["model"]
# params["scale_pos_weight"] = pos_weight 正常数据:异常数据 = 1.15:1

# 3.3 构建数据集
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train  # 保持正负样本比例
)
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

model = lgb.train(params,
                  lgb_train,
                  num_boost_round=config["model"]["num_boost_round"],
                  valid_sets=lgb_val,
                  )

# 4. 保存模型
joblib.dump(model, config["save"]["model_path"])
print("✅ model saved!")

# 5. 模型评估
y_pred = model.predict(X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)

# 5.1 AUC, PR_AUC
auc = roc_auc_score(y_test, y_pred)
pr_auc = average_precision_score(y_test, y_pred)
TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
# 5.2 漏检率
miss_rate = 1 - TP / (TP + FN)
log.info(f"Test_AUC: {auc}")
log.info(f"Test_PR: {pr_auc}")
log.info(f"Miss Rate: {miss_rate}")
log.info(classification_report(y_test, y_pred, target_names=["negative", "positive"]))



