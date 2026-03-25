import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from UAD.GBAE import GB_AE, CenterOnlyAE
from Utils.data_process import *


# ====================== 调用GBAE核心函数 + 保存模型 ======================
def main():
    train_df, test_df = load_kdd_data()
    train_df = preprocess_binary_label(train_df)
    test_df = preprocess_binary_label(test_df)

    # 因为是无监督模型，所以得用正常数据来训练
    train_normal = train_df[train_df["target"] == 0]

    config = load_config()
    print("normal", len(train_normal))
    X_train, y_train, X_test, y_test, y_train_m, y_test_m = preprocess_features(train_normal, test_df, config)

    print('type', type(X_test), type(X_train))
    score_r, score_l, model_state_dict = GB_AE(X_train.to_numpy(), X_test.to_numpy())

    input_dim = X_train.shape[1]
    model = CenterOnlyAE(input_dim=input_dim)
    model.load_state_dict(model_state_dict)

    with open("models//scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # 4. 保存模型
    save_dict = {
        'model_state_dict': model_state_dict,
        'input_dim': input_dim,
        'scaler': scaler,  # 保存归一化器
        'train_anomaly_scores': {'score_r': score_r, 'score_l': score_l},
        'delta': 0.5
    }

    with open('models\\GBAE_trained.pkl', 'wb') as f:
        pickle.dump(save_dict, f)

    print(f"\n✅ 训练完成！模型已保存为 GBAE_trained.pkl")

if __name__ == "__main__":
    main()