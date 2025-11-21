import pandas as pd
import networkx as nx
import random
from gensim.models import Word2Vec
import numpy as np

# 1. 读取CSV并构建图
df = pd.read_csv("肝癌 interact.csv")  # 假设两列：node1,node2
G = nx.from_pandas_edgelist(df, "node1", "node2")

# 2. 定义 Diversified Random Walk
def diversified_random_walk(G, start, length=10):
    walk = [str(start)]
    visited = set([start])
    for _ in range(length - 1):
        cur = int(walk[-1])
        neighbors = list(G.neighbors(cur))
        if neighbors:
            unvisited = [n for n in neighbors if n not in visited]
            if unvisited:
                nxt = random.choice(unvisited)
            else:
                nxt = random.choice(neighbors)
            walk.append(str(nxt))
            visited.add(nxt)
    return walk

# 3. 生成游走序列
walks = []
num_walks = 10  # 每个节点起始的游走次数
walk_length = 20
for node in G.nodes():
    for _ in range(num_walks):
        walks.append(diversified_random_walk(G, node, walk_length))

# 4. 训练 Word2Vec 得到节点嵌入
model = Word2Vec(sentences=walks, size=64, window=5, min_count=0, sg=1, workers=4)
node_embeddings = {str(node): model.wv[str(node)] for node in G.nodes()}

# 5. 提取 link 特征 (Hadamard product)
def get_link_features(u, v, method="hadamard"):
    u_emb = node_embeddings[str(u)]
    v_emb = node_embeddings[str(v)]
    if method == "hadamard":
        return u_emb * v_emb
    elif method == "l1":
        return np.abs(u_emb - v_emb)
    elif method == "l2":
        return (u_emb - v_emb) ** 2
    elif method == "concat":
        return np.concatenate([u_emb, v_emb])
    else:
        raise ValueError("Unknown method")

# 6. 构建新的 DataFrame 并保存
link_features = []
for _, row in df.iterrows():
    u, v = row["node1"], row["node2"]
    feats = get_link_features(u, v, method="hadamard")
    link_features.append([u, v] + feats.tolist())

columns = ["node1", "node2"] + [f"f{i}" for i in range(len(link_features[0]) - 2)]
out_df = pd.DataFrame(link_features, columns=columns)
out_df.to_csv("cancer_link_features.csv", index=False)
