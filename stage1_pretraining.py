import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.io import savemat  # 新增：用于保存.mat文件
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math # 记得在文件顶部加这行！
# =====================
# 超参数设置
# =====================
INPUT_DIM = 400
SIDE = 20
assert SIDE * SIDE == INPUT_DIM
BATCH_SIZE = 256
EPOCHS = 50
LR = 2e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE = 1
FEAT_DIM = 300
EMB_DIM = 300
PROJ_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# =====================
# 数据集
# =====================
class SpectralMultiModalDataset(Dataset):
    def __init__(self, X):
        self.X = X.astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]
        x1d = torch.from_numpy(x).unsqueeze(0)
        x2d = torch.from_numpy(x.reshape(SIDE, SIDE)).unsqueeze(0)
        return x1d, x2d
# =====================
# 1D 模型
# =====================
# ===== 新增：ECA-1D 注意力（替换 SE1D）=====
class ECA1D(nn.Module):
    def __init__(self, channel, k_size=3):
        super().__init__()
        # 自适应选择卷积核大小（经典公式）
        t = int(abs(math.log(channel, 2) + 1) / 2) * 2 + 1 # 保证是奇数
        k_size = k_size if k_size % 2 == 1 else k_size + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x: (B, C, L)
        y = x.mean(dim=-1, keepdim=True) # (B, C, 1) 全局平均
        y = y.transpose(1, 2) # (B, 1, C)
        y = self.conv(y) # 1D 卷积实现跨通道交互
        y = y.transpose(1, 2) # (B, C, 1)
        y = self.sigmoid(y)
        return x * y # 通道加权
# ===== 修改 Block1D：把 SE 换成 ECA =====
class Block1D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 7, stride, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        # 重点：替换 SE → ECA（参数量更少！）
        self.eca = ECA1D(out_c, k_size=5) # k=5 是高光谱最优经验值
        self.down = (nn.Sequential(
            nn.Conv1d(in_c, out_c, 1, stride, bias=False),
            nn.BatchNorm1d(out_c)
        ) if (stride != 1 or in_c != out_c) else nn.Identity())
    def forward(self, x):
        idt = self.down(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.eca(out) # 换成 ECA！
        out = out + idt
        return F.relu(out, inplace=True)
# ===== 最终修改后的 FeatureExtractor1D（只需替换这部分）=====
class FeatureExtractor1D(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, 7, 2, 3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.l1 = Block1D(64, 128, stride=2)
        self.l2 = Block1D(128, 256, stride=2)
        self.l3 = Block1D(256, 256, stride=1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, feat_dim)
    def forward(self, x):
        x = self.stem(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)
# =====================
# 2D 模型
# =====================
class Block2D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.down = (nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c))
                     if (stride != 1 or in_c != out_c) else nn.Identity())
    def forward(self, x):
        idt = self.down(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + idt, inplace=True)
class FeatureExtractor2D(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.l1 = Block2D(32, 64, stride=2)
        self.l2 = Block2D(64, 128, stride=2)
        self.l3 = Block2D(128, 256, stride=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, feat_dim)
    def forward(self, x):
        x = self.stem(x)
        x = self.l1(x); x = self.l2(x); x = self.l3(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return self.fc(x)
# =====================
# 共享编码器
# =====================
class PairEncoder(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, emb_dim=EMB_DIM, nhead=4, nlayers=2, ff=1024):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, feat_dim))
        self.pos = nn.Parameter(torch.zeros(1, 3, feat_dim))
        layer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=nhead,
                                           dim_feedforward=ff, batch_first=True, dropout=0.1, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.proj = nn.Linear(feat_dim, emb_dim)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
    def forward(self, f1d, f2d):
        B = f1d.size(0)
        cls = self.cls.expand(B, -1, -1)
        tokens = torch.cat([cls, f1d.unsqueeze(1), f2d.unsqueeze(1)], dim=1)
        tokens = tokens + self.pos
        enc = self.encoder(tokens)
        cls_emb = enc[:, 0]
        return self.proj(cls_emb)
# =====================
# 投影头 + 损失
# =====================
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=PROJ_DIM):
        super().__init__()
        hid = max(in_dim, proj_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.BatchNorm1d(hid), nn.ReLU(inplace=True),
            nn.Linear(hid, proj_dim)
        )
    def forward(self, x): return self.net(x)
def nt_xent_loss(z_a, z_b, temperature=TEMPERATURE):
    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)
    B = z_a.size(0)
    z = torch.cat([z_a, z_b], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2*B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, -9e15)
    pos = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(z.device)
    log_prob = F.log_softmax(sim, dim=1)
    return -log_prob[torch.arange(2*B), pos].mean()
# =====================
# 预训练主函数
# =====================
def pretrain_multimodal(X_train):
    ds = SpectralMultiModalDataset(X_train)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    fe1d = FeatureExtractor1D(FEAT_DIM).to(DEVICE)
    fe2d = FeatureExtractor2D(FEAT_DIM).to(DEVICE)
    pair_enc = PairEncoder(FEAT_DIM, EMB_DIM).to(DEVICE)
    proj_uni = ProjectionHead(FEAT_DIM, PROJ_DIM).to(DEVICE)
    proj_joint = ProjectionHead(EMB_DIM, PROJ_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(
        list(fe1d.parameters()) + list(fe2d.parameters()) +
        list(pair_enc.parameters()) + list(proj_uni.parameters()) +
        list(proj_joint.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    loss_curve = []
    for epoch in range(1, EPOCHS + 1):
        fe1d.train(); fe2d.train(); pair_enc.train()
        proj_uni.train(); proj_joint.train()
        running_loss = 0.0
        for x1d, x2d in loader:
            x1d, x2d = x1d.to(DEVICE), x2d.to(DEVICE)
            f1d = fe1d(x1d)
            f2d = fe2d(x2d)
            joint = pair_enc(f1d, f2d)
            z1 = proj_uni(f1d)
            z2 = proj_uni(f2d)
            zj = proj_joint(joint)
            loss_12 = nt_xent_loss(z1, z2)
            loss_j1 = nt_xent_loss(zj, z1)
            loss_j2 = nt_xent_loss(zj, z2)
            loss = 0.5 * loss_12 + 0.5 * (loss_j1 + loss_j2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(loader)
        loss_curve.append(avg_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{EPOCHS}] Loss: {avg_loss:.4f}")
    # =====================
    # 保存权重（关键！）
    # =====================
    torch.save({
        'fe1d': fe1d.state_dict(),
        'fe2d': fe2d.state_dict(),
        'pair_enc': pair_enc.state_dict(),
        'proj_uni': proj_uni.state_dict(),
        'proj_joint': proj_joint.state_dict(),
        'epoch': epoch,
        'loss': loss_curve[-1],
        'loss_curve': loss_curve
    }, "mm_contrastive_best.pth")
    print("Success: 完整模型已保存到 mm_contrastive_best.pth")
    # 单独保存 fe2d
    torch.save(fe2d.state_dict(), "fe2d_extractor_only.pth")
    print("Success: fe2d 单独权重已保存到 fe2d_extractor_only.pth")
    # =====================
    # 可视化（修复设备问题）
    # =====================
    fe1d.eval(); fe2d.eval(); pair_enc.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        x1d, x2d = batch[0].to(DEVICE), batch[1].to(DEVICE)
        f1 = fe1d(x1d) # (B, 300) GPU
        f2 = fe2d(x2d) # (B, 300) GPU
        j = pair_enc(f1, f2) # (B, 300) GPU
        # 转 CPU → numpy
        f1_np = f1.cpu().numpy()
        f2_np = f2.cpu().numpy()
        j_np = j.cpu().numpy()
        # t-SNE 可视化 + 保存数值
        emb, y = visualize_tsne(
            f1_np,
            f2_np,
            j_np,
            f"Final Embeddings (Epoch {EPOCHS})",
            "emb_final.png"
        )
        # 保存 t-SNE 数值到 .mat（当前模型为最佳）
        mat_data = {
            'tsne_emb': emb,
            'labels': y
        }
        savemat("final_tsne.mat", mat_data)
        print("t-SNE 数值已保存到 final_tsne.mat")
    # 损失曲线
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(loss_curve, lw=2, color='tab:blue')
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Multimodal Contrastive Pretraining")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pretrain_loss.png", dpi=180)
    plt.show()
    return fe1d, fe2d, pair_enc, proj_uni, proj_joint
# =====================
# t-SNE 可视化（修改版：返回emb和y，便于保存数值）
# =====================
def visualize_tsne(feat1d, feat2d, joint, title, save_path=None):
    n = min(feat1d.shape[0], feat2d.shape[0], joint.shape[0], 300)
    X = np.vstack([feat1d[:n], feat2d[:n], joint[:n]])
    y = np.array([0]*n + [1]*n + [2]*n)
    tsne = TSNE(n_components=2, perplexity=min(30, n//3), random_state=SEED, init="pca")
    emb = tsne.fit_transform(X)
    plt.figure(figsize=(6.4, 5.2))
    plt.scatter(emb[y==0,0], emb[y==0,1], s=14, alpha=0.7, label='1D Feature', color='tab:blue')
    plt.scatter(emb[y==1,0], emb[y==1,1], s=14, alpha=0.7, label='2D Feature', color='tab:orange')
    plt.scatter(emb[y==2,0], emb[y==2,1], s=14, alpha=0.7, label='Joint CLS', color='tab:green')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180)
    plt.show()
    return emb, y  # 新增：返回t-SNE坐标和标签
# =====================
# 主函数
# =====================
def main():
    file_path = "T_Spectr_Maize.xlsx"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到数据文件: {file_path}")
    df = pd.read_excel(file_path)
    X = df.iloc[:, 2:].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    print("开始多模态对比预训练（1D + 2D → 共享 Transformer）...")
    pretrain_multimodal(Xs)
if __name__ == "__main__":
    main()