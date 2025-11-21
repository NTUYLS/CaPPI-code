import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ========= 交叉注意力模块 =========
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=256, num_heads=4, target_dim=256, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.attn_a2b = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_b2a = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.ln_a2b = nn.LayerNorm(dim)
        self.ffn_a2b = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )
        self.ln_b2a = nn.LayerNorm(dim)
        self.ffn_b2a = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )
        self.proj = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, target_dim))

    def forward(self, A, B):
        if A.dim() == 2: A = A.unsqueeze(1)
        if B.dim() == 2: B = B.unsqueeze(1)

        a2b, _ = self.attn_a2b(A, B, B, need_weights=False)
        a2b = self.ln_a2b(A + a2b)
        a2b = self.ln_a2b(a2b + self.ffn_a2b(a2b))

        b2a, _ = self.attn_b2a(B, A, A, need_weights=False)
        b2a = self.ln_b2a(B + b2a)
        b2a = self.ln_b2a(b2a + self.ffn_b2a(b2a))

        a2b_pooled = a2b.mean(dim=1)
        b2a_pooled = b2a.mean(dim=1)
        fused = torch.cat([a2b_pooled, b2a_pooled], dim=-1)
        return self.proj(fused)


# ========= 主程序 =========
def main():
    in_a, in_b, out_path = "Gated_256.csv", "cancer_Behavior_feature_256_40.csv", "fused_features.csv"

    # 强制不把第一行当作header
    df1 = pd.read_csv(in_a, header=None)
    df2 = pd.read_csv(in_b, header=None)

    print("原始行数: df1 =", len(df1), ", df2 =", len(df2))

    # 取最大长度，保证对齐
    max_len = max(len(df1), len(df2))

    # 如果某个文件行数不足，补零
    if len(df1) < max_len:
        pad = pd.DataFrame(np.zeros((max_len - len(df1), df1.shape[1])))
        df1 = pd.concat([df1, pad], ignore_index=True)

    if len(df2) < max_len:
        pad = pd.DataFrame(np.zeros((max_len - len(df2), df2.shape[1])))
        df2 = pd.concat([df2, pad], ignore_index=True)

    # 现在一定是相同长度
    print("对齐后行数: df1 =", len(df1), ", df2 =", len(df2))

    id_vals = np.arange(max_len)
    A, B = df1.to_numpy(), df2.to_numpy()

    # 转 tensor
    A_t = torch.tensor(A, dtype=torch.float32)
    B_t = torch.tensor(B, dtype=torch.float32)

    # 融合
    model = CrossAttentionFusion(dim=256, num_heads=4, target_dim=256)
    with torch.no_grad():
        fused = model(A_t, B_t).numpy()

    # 保存
    out_df = pd.DataFrame(fused, columns=[f"f{i+1}" for i in range(fused.shape[1])])
    out_df.insert(0, "id", id_vals)
    out_df.to_csv(out_path, index=False)

    print(f"融合完成：输入 A={len(df1)} 行, B={len(df2)} 行；输出 {len(out_df)} 行 -> {out_path}")


if __name__ == "__main__":
    main()
