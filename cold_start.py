import numpy as np
import pandas as pd
import pickle
import torch
from UAD.GBAE import CenterOnlyAE

def load_gbae_model(model_path="GBAE_trained.pkl"):
    # 加载模型文件
    with open(model_path, "rb") as f:
        model_dict = pickle.load(f)

    # 重构模型
    model = CenterOnlyAE(input_dim=model_dict["input_dim"])
    model.load_state_dict(model_dict["model_state_dict"])
    model.eval()

    scaler = model_dict["scaler"]
    return model, scaler

def predict_cold_start(model, scaler, X):
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        recon, _ = model(X_tensor)

    anomaly_scores = torch.mean((recon - X_tensor) ** 2, dim=1).numpy()

    return anomaly_scores

if __name__ == "__main__":
    model, scaler = load_gbae_model()

    df = pd.read_csv("cold_start_data.csv")  # 你自己的无标签数据
    X = df.to_numpy()
    scores = predict_cold_start(model, scaler, X)


    threshold = np.percentile(scores, 95)  # 取前 5% 最异常的
    pred_labels = (scores > threshold).astype(int)

    df["anomaly_score"] = scores
    df["pred_label"] = pred_labels
    df.to_csv("cold_start_result.csv", index=False)

    print("✅ 冷启动检测完成！结果已保存：cold_start_result.csv")
    print(f"异常数量：{sum(pred_labels)}")