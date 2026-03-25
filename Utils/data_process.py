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


# 2. 两层标签
def make_labels(df):
    # 1.二分类
    df["target"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

    # ===================== 【正确】KDD 攻击类型自动归类 =====================
    def map_attack(attack_name):
        # 1. DoS 暴力拒绝服务
        dos = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'apache2', 'mailbomb', 'processtable', 'udpstorm']
        # 2. Probe 扫描探测
        probe = ['satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint']
        # 3. U2R 普通用户提权
        u2r = ['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps']
        # 4. R2L 远程未授权访问
        r2l = ['ftp_write', 'guess_passwd', 'imap', 'phf', 'spy', 'warezclient', 'warezmaster', 'xlock', 'xsnoop',
               'snmpguess', 'snmpgetattack', 'httptunnel', 'sendmail', 'named']

        if attack_name == 'normal':
            return 0
        elif attack_name in dos:
            return 1
        elif attack_name in probe:
            return 2
        elif attack_name in u2r:
            return 3
        elif attack_name in r2l:
            return 4
        else:
            return 0  # 未知攻击 → 归为正常（不会出NaN）

    df["multi_target"] = df["label"].apply(map_attack)

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
    train_df = make_labels(train_df)
    test_df = make_labels(test_df)

    drop_feat = ["target", "difficulty_score","label","multi_target"]
    X_train = train_df.drop(columns=drop_feat, axis=1)
    X_test = test_df.drop(columns=drop_feat, axis=1)
    y_train = train_df["target"]
    y_test = test_df["target"]
    y_train_m = train_df["multi_target"]
    y_test_m = test_df["multi_target"]

    encoder_path = "./models/OrdinalEncoder.pkl"
    if os.path.exists(encoder_path):
        # 已存在 → 直接加载
        encoder = joblib.load(encoder_path)
    else:
        # 不存在 → 用全量训练集训练一次（保证见过所有类别）
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(train_df[CAT_Features])
        joblib.dump(encoder, encoder_path)

    X_train[CAT_Features] = encoder.transform(X_train[CAT_Features])
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

    return X_train, y_train, X_test, y_test, y_train_m, y_test_m



