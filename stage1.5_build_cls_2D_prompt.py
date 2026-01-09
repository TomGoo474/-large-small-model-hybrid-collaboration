# 1_generate_prompt_dataset.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import pickle

FEAT_DIM = 300

# ===================== 正确的 ECA1D（固定 k=5，和原模型完全一致）=====================
class ECA1D(nn.Module):
    def __init__(self, channel, k_size=5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        y = x.mean(dim=-1, keepdim=True)          # (B, C, 1)
        y = y.transpose(1, 2)                     # (B, 1, C)
        y = self.conv(y)                          # (B, 1, C)
        y = y.transpose(1, 2)                     # (B, C, 1)
        y = self.sigmoid(y)
        return x * y


# ===================== 1D 特征提取器（全部改回 BatchNorm1d）=====================
class Block1D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 7, stride, 3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_c)          # ← 1d
        self.conv2 = nn.Conv1d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_c)          # ← 1d（关键修复）

        self.eca = ECA1D(out_c, k_size=5)

        self.down = nn.Identity()
        if stride != 1 or in_c != out_c:
            self.down = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm1d(out_c)
            )

    def forward(self, x):
        identity = self.down(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.eca(out)
        out = out + identity
        return F.relu(out, inplace=True)


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
        x = self.gap(x)              # (B, 256, 1)
        x = x.flatten(1)             # (B, 256)
        return self.fc(x)


# ===================== 2D 特征提取器（保持不变，但确保 bn2 是 2d）=====================
class Block2D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)

        self.down = nn.Identity()
        if stride != 1 or in_c != out_c:
            self.down = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        identity = self.down(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out, inplace=True)


class FeatureExtractor2D(nn.Module):
    def __init__(self, feat_dim=300):
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
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.gap(x)              # (B, 256, 1, 1)
        x = x.flatten(1)             # (B, 256)
        return self.fc(x)


# ===================== Pair Encoder（不变）=====================
class PairEncoder(nn.Module):
    def __init__(self, feat_dim=300, emb_dim=300, nhead=4, nlayers=2, ff=1024):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, feat_dim))
        self.pos = nn.Parameter(torch.zeros(1, 3, feat_dim))
        layer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=nhead,
                                           dim_feedforward=ff, batch_first=True,
                                           dropout=0.1, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.proj = nn.Linear(feat_dim, emb_dim)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, f1d, f2d):
        B = f1d.size(0)
        cls = self.cls.expand(B, -1, -1)
        tokens = torch.cat([cls, f1d.unsqueeze(1), f2d.unsqueeze(1)], dim=1)
        tokens = tokens + self.pos.expand(B, -1, -1)
        enc = self.encoder(tokens)
        return self.proj(enc[:, 0])


# ===================== Prompt 构造函数（不变）=====================
def create_t2mfdf_prompt(raw_spectrum_2d, cls_feature_vec, center_size=20, border_scale=0.3):
    assert raw_spectrum_2d.shape == (center_size, center_size)
    assert cls_feature_vec.shape == (300,)

    center = raw_spectrum_2d.copy()
    center = (center - center.mean()) / (center.std() + 1e-8)
    center = np.clip(center, -3, 3)

    feat_mat = cls_feature_vec.reshape(10, 30)
    feat_mat = (feat_mat - feat_mat.mean()) / (feat_mat.std() + 1e-8)
    feat_mat *= border_scale

    left   = np.rot90(feat_mat, k=1)
    right  = np.rot90(feat_mat, k=3)
    top    = feat_mat.copy()
    bottom = np.rot90(feat_mat, k=2)

    H = W = center_size + 20
    prompt = np.zeros((H, W), dtype=np.float32)

    prompt[0:10,   10:40] = top
    prompt[30:40,   0:30] = bottom
    prompt[0:30,    0:10] = left
    prompt[10:40,  30:40] = right
    prompt[10:30,  10:30] = center

    return prompt


# ===================== 主函数（已完整保存 ID）=====================
def generate_prompt_dataset():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_file = "train_set_with_id1.xlsx"
    test_file  = "val_set_with_id1.xlsx"
    model_path = "mm_contrastive_best.pth"
    output_pkl = "t2mfdf_prompt_dataset_with_id.pkl"

    for f in [train_file, test_file, model_path]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"文件不存在: {f}")

    # 模型定义
    fe1d = FeatureExtractor1D().to(device)
    fe2d = FeatureExtractor2D().to(device)
    pair_enc = PairEncoder().to(device)

    # 加载权重
    ckpt = torch.load(model_path, map_location=device)
    fe1d.load_state_dict(ckpt['fe1d'])
    fe2d.load_state_dict(ckpt['fe2d'])
    pair_enc.load_state_dict(ckpt['pair_enc'])
    print("预训练模型加载成功")

    fe1d.eval()
    fe2d.eval()
    pair_enc.eval()

    # 读取数据
    df_train = pd.read_excel(train_file)
    df_test  = pd.read_excel(test_file)

    train_ids = df_train['ID'].astype(str).values
    test_ids  = df_test['ID'].astype(str).values

    X_train = df_train.iloc[:, 2:].values.astype(np.float32)
    y_train = df_train['Class'].map({'Non': 0, 'AF': 1}).values

    X_test  = df_test.iloc[:, 2:].values.astype(np.float32)
    y_test  = df_test['Class'].map({'Non': 0, 'AF': 1}).values

    assert X_train.shape[1] == 400 and X_test.shape[1] == 400

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.pkl")

    # 批量提取 + 生成 Prompt + 保存 ID
    def process_set(X_scaled, y, ids, name):
        prompts, labels, id_list = [], [], []
        batch_size = 64

        for i in range(0, len(X_scaled), batch_size):
            batch_X = X_scaled[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            batch_id = ids[i:i+batch_size]

            spec_1d = torch.from_numpy(batch_X).unsqueeze(1).to(device)          # (B,1,400)
            spec_2d = spec_1d.view(-1, 1, 20, 20)                                # (B,1,20,20)

            with torch.no_grad():
                f1d = fe1d(spec_1d)
                f2d = fe2d(spec_2d)
                cls_emb = pair_enc(f1d, f2d)                                    # (B,300)

            cls_np = cls_emb.cpu().numpy()

            for j in range(len(batch_X)):
                raw_2d = batch_X[j].reshape(20, 20)
                prompt = create_t2mfdf_prompt(raw_2d, cls_np[j])
                prompts.append(prompt)
                labels.append(batch_y[j])
                id_list.append(batch_id[j])

            print(f"{name} processed: {min(i+batch_size, len(X_scaled))}/{len(X_scaled)}", end="\r")

        print()
        return (np.stack(prompts).astype(np.float32),
                np.array(labels, dtype=np.int64),
                np.array(id_list))

    print("正在生成训练集 Prompt...")
    train_prompts, train_labels, train_ids_final = process_set(X_train_scaled, y_train, train_ids, "Train")

    print("正在生成测试集 Prompt...")
    test_prompts,  test_labels,  test_ids_final  = process_set(X_test_scaled,  y_test,  test_ids,  "Test")

    dataset = {
        'train_prompts': train_prompts,
        'train_labels':  train_labels,
        'train_ids':     train_ids_final,
        'test_prompts':  test_prompts,
        'test_labels':   test_labels,
        'test_ids':      test_ids_final,
        'scaler':        scaler
    }

    with open(output_pkl, "wb") as f:
        pickle.dump(dataset, f)

    print("\n全部完成！")
    print(f"保存文件: {output_pkl}")
    print(f"训练集样本数: {len(train_labels)}  (ID数: {len(train_ids_final)})")
    print(f"测试集样本数: {len(test_labels)}   (ID数: {len(test_ids_final)})")

if __name__ == "__main__":
    generate_prompt_dataset()