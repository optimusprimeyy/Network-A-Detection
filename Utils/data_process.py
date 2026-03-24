import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import json
columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate", "srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","attack", "last_flag"
]

CAT_Features = ["protocol_type", "service", "flag"]

# 加载JSON配置文件
def load_config(config_path="./config/params.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    print("✅ 配置文件加载成功")
    return config

# 加载kdd数据集
def load_kdd_data():
    train_path = "datasets//Train_with_target.csv"
    test_path = "datasets//Test_with_target.csv"
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    print("✅ 数据加载完成")

    return train_data, test_data

# 2：生成二分类标签（正常/异常）
def preprocess_binary_label(df):
    """normal = 0, anomaly = 1"""
    df["target"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)
    return df

# 3. 异常值处理(分位数缩尾法: 平衡极端值带来的干扰，又保留了attack特征)
def handle_anomalies(train_data, test_data, numerical_features, config):
    """
    对数值特征做1%-99%缩尾处理，截断极端异常值。测试集也要复用！
    """
    lower = config["preprocess"]["anomaly_lower"]
    upper = config["preprocess"]["anomaly_upper"]
    for col in numerical_features:
        lower_bound = train_data[col].quantile(lower)
        upper_bound = train_data[col].quantile(upper)

        # 截断
        train_data[col] = np.clip(train_data[col], lower_bound, upper_bound)
        test_data[col] = np.clip(test_data[col], lower_bound, upper_bound)

    print("✅ 异常值处理完成")

    return train_data, test_data

# 4. 特征预处理（编码+标准化）
def preprocess_features(train_df, test_df, config, save_path = "./models"):
    """
    category_feature: Label_Encoder编码
    数值特征 StandardScaler标准化
    """
    drop_feat = ["target", "difficulty_score","label"]
    X_train = train_df.drop(columns=drop_feat, axis=1)
    X_test = test_df.drop(columns=drop_feat, axis=1)
    y_train = train_df["target"]
    y_test = test_df["target"]

    # 分类特征编码
    encoder = OrdinalEncoder()
    X_train[CAT_Features] = encoder.fit_transform(X_train[CAT_Features])
    X_test[CAT_Features] = encoder.transform(X_test[CAT_Features])

    # 异常值处理
    NUM_features = [feat for feat in X_train.columns if feat not in CAT_Features]
    X_train, X_test = handle_anomalies(X_train, X_test, NUM_features, config)

    # 数值特征标准化
    scaler = StandardScaler()
    X_train[NUM_features] = scaler.fit_transform(X_train[NUM_features])
    X_test[NUM_features] = scaler.transform(X_test[NUM_features])

    # 保存标准化器
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(scaler, os.path.join(save_path, "scaler.pkl"))
    print("✅ 特征预处理完成")

    return X_train, y_train, X_test, y_test



