import h5py
import numpy as np

# 使用正确的路径格式
features = {
    "p1/s1/d1": np.random.randn(512).astype(np.float32),  # 512维特征
    "p2/s2/d2": np.random.randn(512).astype(np.float32),
    "p3/s3/d3": np.random.randn(512).astype(np.float32)
}

# 投影特征（用于检索）
projections = {
    "p1/s1/d1": np.random.randn(128).astype(np.float32),  # 128维投影
    "p2/s2/d2": np.random.randn(128).astype(np.float32),
    "p3/s3/d3": np.random.randn(128).astype(np.float32)
}

with h5py.File('features.h5', 'w') as f:
    # 创建两个组：img_embed 和 img_proj
    for path, feat in features.items():
        f.create_dataset(f'img_embed/{path}', data=feat)

    for path, proj in projections.items():
        f.create_dataset(f'img_proj/{path}', data=proj)

print("✅ features.h5 已生成（修正格式）！")