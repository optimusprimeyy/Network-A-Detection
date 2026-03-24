import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors  # 导入 NearestNeighbors
from .GBshengcheng_v2 import getGranularBall  # 假设这个模块存在

# 设置几何一致性约束的平衡因子
LAMBDA_GC_CENTER = 0.01

# 稳健归一化示例（不受极端值影响）
def robust_norm(x):
    """通过中位数和四分位距进行稳健归一化。"""
    median = np.median(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    # 避免除以零
    return (x - median) / (iqr + 1e-10)


# -------------------------- 1. 粒球类（仅含原始中心） --------------------------
class GranularBall:
    def __init__(self, data,
                 index):  # data的倒数第二列是label，最后一列是index
        self.data = data[:, :-1]  # 提取特征数据
        self.index = index
        self.center = self.data.mean(0)  # 计算粒球中心
        self.score = 0
        self.radius = self.calculate_radius()  # 计算半径

    def calculate_radius(self):
        # 计算粒球的半径，这里使用数据点到中心点距离的最大值作为半径
        distances = np.sqrt(np.sum((self.data - self.center) ** 2, axis=1))
        radius = np.max(distances) if len(self.data) > 1 else 1e-6  # 避免半径为0
        return radius


# -------------------------- 3. 自编码器模型（适配小数据） --------------------------
class CenterOnlyAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, dropout_rate=0.3):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(2, input_dim // 4)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z  # 返回重构结果和潜在特征


# -------------------------- 4. 模型训练（仅用粒球中心, 包含中心几何一致性损失） --------------------------
def train_center_only_model(center_data, epochs=100, batch_size=32, hidden_dim=None, patience=15):
    """用粒球中心训练AE，使用中心几何一致性约束 L_GC_Center"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = center_data.shape[1]
    n_centers = len(center_data)
    batch_size = min(batch_size, n_centers)

    # --- 预计算中心几何一致性锚点 ---
    # nn_center_space = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(center_data)
    # distances_raw, indices = nn_center_space.kneighbors(center_data)

    # center_dists_orig = distances_raw[:, 1] #[:,0]是与自己的距离，全为0
    # nearest_center_indices = indices[:, 1] # [:,0]是自己，0,1,2,3,4

    center_tensor = torch.tensor(center_data, dtype=torch.float32).to(device)
    # center_dists_orig_tensor = torch.tensor(center_dists_orig, dtype=torch.float32).to(device)
    # nearest_center_indices_tensor = torch.tensor(nearest_center_indices, dtype=torch.long).to(device)


    model = CenterOnlyAE(input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    print(f"训练配置：粒球中心数={n_centers}, batch_size={batch_size}, 设备={device}")
    train_losses = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n_centers)
        epoch_loss = 0.0

        for i in range(0, n_centers, batch_size):
            optimizer.zero_grad()
            batch_indices = permutation[i:i + batch_size]
            batch_c = center_tensor[batch_indices]

            # batch_dists_orig = center_dists_orig_tensor[batch_indices]
            # batch_nearest_center_indices = nearest_center_indices_tensor[batch_indices]

            recon_c, z_c = model(batch_c)

            # --- 获取 z_k* ---
            model.eval()
            with torch.no_grad():
                _, Z_all = model(center_tensor)
            model.train()
            # z_nearest_center = Z_all[batch_nearest_center_indices]
            # ------------------

            # 1. 重构损失 (L2 Loss)
            recon_loss = torch.mean((recon_c - batch_c) ** 2)

            # 3. 总损失
            total_loss = recon_loss

            # 注意：未包含梯度裁剪，如果出现 NaN，需要手动处理
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item() * len(batch_c)

        epoch_loss /= n_centers
        train_losses.append(epoch_loss)

        # 早停检查
        early_stopping(epoch_loss)
        if early_stopping.early_stop:
            print(f"早停触发，停止于Epoch {epoch + 1}")
            return model, train_losses

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], 总损失: {epoch_loss:.6f}")

    return model, train_losses


# -------------------------- 5. 异常分数计算（重构误差 + 潜在空间一致性误差） --------------------------
def compute_anomaly_scores(samples, model, gb_list):
    """
    计算综合异常分数：重构误差 (Score_R) + 潜在空间一致性误差 (Score_L)
    Score_L = | ||x_i - c_j*|| - ||z_i - z_j*|| |
    """
    device = next(model.parameters()).device
    model.eval()
    sample_tensor = torch.tensor(samples, dtype=torch.float32).to(device)

    # 提取所有中心信息
    raw_centers = np.array([gb.center for gb in gb_list])
    if len(raw_centers) == 0:
        return np.zeros(len(samples))

    # 找到每个样本最近的中心 c_j*
    nn_finder = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(raw_centers)
    distances_to_centers, nearest_center_indices = nn_finder.kneighbors(samples)
    nearest_center_indices = nearest_center_indices.flatten()

    # 原始空间距离 ||x_i - c_j*||_2
    orig_dists = distances_to_centers.flatten()

    with torch.no_grad():
        # Z_samples 是 z_i
        recon, Z_samples = model(sample_tensor)
        recon_error = torch.mean((recon - sample_tensor) ** 2, dim=1).cpu().numpy()

        # 提取所有中心 c_j 的潜在特征 Z_all
        center_tensor = torch.tensor(raw_centers, dtype=torch.float32).to(device)
        _, Z_all = model(center_tensor)

        # 提取最近中心 c_j* 的潜在特征 z_j*
        Z_nearest_centers = Z_all[nearest_center_indices]

        # 2. 潜在空间一致性误差 (Score_L)
    # 潜在空间距离 ||z_i - z_j*||_2
    latent_distances = torch.sqrt(torch.sum((Z_samples - Z_nearest_centers) ** 2, dim=1)).cpu().numpy()

    # Score_L = | 原始空间距离 - 潜在空间距离 |
    consistency_error = np.abs(orig_dists - latent_distances)
    # consistency_error = latent_distances / (orig_dists + 1e-5)

    # 3. 归一化融合
    recon_norm = robust_norm(recon_error)
    consistency_norm = robust_norm(consistency_error)


    # 再次归一化综合分数，确保其分布良好
    return recon_norm, consistency_norm


# -------------------------- 6. 早停机制（保持不变） --------------------------
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=1e-6):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def add_center(gb_list):
    """从原始粒球数据列表创建 GranularBall 实例和中心数据数组"""
    gb_dist = []
    center_data = []
    for i in range(0, len(gb_list)):
        gb = GranularBall(gb_list[i], i)
        gb_dist.append(gb)
        center_data.append(gb.center)
    center_data = np.array(center_data)
    return gb_dist, center_data


def GB_AE(X_train, X_test):
    """在线进化型模糊粒球（OE-GB）引导的AE异常检测"""
    # 1. 初始化粒球（中心直接作为训练数据）
    gb_list_raw = getGranularBall(X_train, 0.5)
    gb_list, center_data = add_center(gb_list_raw)  # 封装成一个一个类


    # 2. 仅用粒球中心训练模型
    model, train_losses = train_center_only_model(
        center_data=center_data,
        epochs=200,
        batch_size=32,
    )

    # 5. 计算异常分数 (综合 Score_R + Score_L)
    Score_R, Score_L= compute_anomaly_scores(
        samples = X_test,
        model = model,
        gb_list = gb_list  # 传入粒球列表用于计算潜在空间一致性误差
    )

    return Score_R,  Score_L, model.state_dict()