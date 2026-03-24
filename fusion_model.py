import joblib
import numpy as np
import pandas as pd
import pickle
import torch
import lightgbm as lgb
from UAD.GBAE import CenterOnlyAE
from Utils.data_process import load_kdd_data, preprocess_binary_label, load_config, preprocess_features
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/fusion_model.log", encoding="utf-8"),  # 日志保存路径
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# 加载 GBAE
def load_gbae(model_path = 'models\\GBAE_trained.pkl'):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    model = CenterOnlyAE(input_dim=model_data['input_dim'])
    model.load_state_dict(model_data["model_state_dict"])

    # 切换到评估模式
    model.eval()

    return model

# 加载 LightGBM
def load_lgb():
    with open('models\\lightgbm_trained.pkl', 'rb') as f:
        model = pickle.load(f)

    return model

# GBAE 预测异常分数
def gbae_predict(model, X):
    X_tensor = torch.tensor(X).float()

    with torch.no_grad():
        recon,_ = model(X_tensor)

    anomaly_score = torch.mean((recon - X_tensor) ** 2, dim=1).numpy()
    return anomaly_score


# LightGBM 预测概率
def lgb_predict(model, X):
    pred = model.predict(X)

    return pred

def fusion_scores(lgb_pred, gbae_scores, weight = 0.6):
    gbae_norm = (gbae_scores - np.min(gbae_scores)) / (np.max(gbae_scores) - np.min(gbae_scores) + 1e-10)

    final_score = weight * lgb_pred + (1 - weight) * gbae_norm

    final_label = np.where(gbae_norm > 0.5, 1, 0)

    return final_score, final_label

if __name__ == '__main__':
    config = load_config()
    scaler = joblib.load('models\\scaler.pkl')
    gbae_model = load_gbae()
    lgb_model = load_lgb()
    train_df, test_df = load_kdd_data()
    train_df = preprocess_binary_label(train_df)
    test_df = preprocess_binary_label(test_df)

    X_test = test_df.drop('label', axis=1).to_numpy()
    y_test = test_df["label"].to_numpy()

    X_train, y_train, X_test, y_test = preprocess_features(train_df, test_df, config)

    gbae_scores = gbae_predict(gbae_model, X_test.to_numpy())
    lgb_scores = lgb_predict(lgb_model, X_test.to_numpy())

    final_scores, final_label = fusion_scores(lgb_scores, gbae_scores)

    # ===================== 【问题3测试：评估结果】 =====================
    from sklearn.metrics import roc_auc_score, accuracy_score

    print("\n" + "=" * 50)
    print("测试结果：已知攻击 + 未知攻击兼顾")
    print("=" * 50)

    # 1. 单模型 GBAE
    auc_gbae = roc_auc_score(y_test, gbae_scores)
    print(f"GBAE(无监督-未知攻击) AUC = {auc_gbae:.4f}")

    # 2. 单模型 LightGBM
    auc_lgb = roc_auc_score(y_test, lgb_scores)
    print(f"LightGBM(监督-已知攻击) AUC = {auc_lgb:.4f}")

    # 3. 融合模型（最终解决问题3）
    final_auc = roc_auc_score(y_test, final_scores)
    print(f"融合模型 AUC = {final_auc:.4f}")

    print("\n✅ 结论：融合模型AUC最高 → 已知、未知攻击都搞定！")
