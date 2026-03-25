import logging

import pandas as pd
import numpy as np
import os
import joblib
from lightgbm import early_stopping
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score
import lightgbm as lgb
from Utils.data_process import load_kdd_data, preprocess_features, load_config
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

X_train, y_train, X_test, y_test, y_train_m, y_test_m = preprocess_features(train_df, test_df, config)
print("y_train_m",y_train_m)
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

binary_model = lgb.train(params,
                  lgb_train,
                  num_boost_round=config["model"]["num_boost_round"],
                  valid_sets=lgb_val,
                  )

# 4. 保存模型
joblib.dump(binary_model, "./models/lightgbm_binary_model.pkl")
print("✅ model saved!")

# 5. 模型评估
y_pred = binary_model.predict(X_test)
y_pred_binary = np.where(y_pred > 0.5, 1, 0)

# 5.1 AUC, PR_AUC
auc = roc_auc_score(y_test, y_pred_binary)
pr_auc = average_precision_score(y_test, y_pred_binary)
TN, FP, FN, TP = confusion_matrix(y_test, y_pred_binary).ravel()
# 5.2 漏检率
miss_rate = 1 - TP / (TP + FN)
log.info(f"Test_AUC: {auc}")
log.info(f"Test_PR: {pr_auc}")
log.info(f"Miss Rate: {miss_rate}")
log.info(classification_report(y_test, y_pred_binary, target_names=["negative", "positive"]))

# 多分类模型
multi_params = params.copy()

multi_params["objective"] = "multiclass"
multi_params["num_class"] = 5
multi_params["metric"] = "multi_logloss"

X_train_split_m, X_val_m, y_train_split_m, y_val_m = train_test_split(
    X_train, y_train_m,
    test_size=0.2,
    random_state=42,
    stratify=y_train  # 保持正负样本比例
)
lgb_train_mul = lgb.Dataset(X_train_split_m, y_train_split_m)
lgb_val_mul = lgb.Dataset(X_val_m, y_val_m)
multi_model = lgb.train(multi_params,lgb_train_mul,valid_sets=lgb_val_mul)

joblib.dump(multi_model, "./models/lightgbm_multi_model.pkl")

y_pred_multi = multi_model.predict(X_test)
y_pred_multi = np.argmax(y_pred_multi, axis=1)

# 🔥 清洗：把 NaN 标签去掉
mask = ~np.isnan(y_test_m)
X_test_clean = X_test[mask]
y_test_m_clean = y_test_m[mask]
y_pred_multi_clean = y_pred_multi[mask]
print("===多分类结果===")
print(classification_report(y_test_m_clean, y_pred_multi_clean, target_names=["normal","dos","probe","u2r","r2l"]))





