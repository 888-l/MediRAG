import h5py
import numpy as np

# 模拟图像特征：使用 BioViL-T 的典型输出维度 (1, 197, 768)
# 键名为 dicom_id
features = {
    "d1": np.random.randn(1, 197, 768).astype(np.float32),
    "d2": np.random.randn(1, 197, 768).astype(np.float32),
    "d3": np.random.randn(1, 197, 768).astype(np.float32)
}

with h5py.File('features.h5', 'w') as f:
    for img_id, feat in features.items():
        f.create_dataset(img_id, data=feat)

print("✅ features.h5 已生成！")