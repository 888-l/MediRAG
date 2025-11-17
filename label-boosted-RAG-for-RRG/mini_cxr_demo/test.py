import pandas as pd
import h5py
import numpy as np


def validate_dataset():
    """验证数据集格式是否正确"""

    # 检查CSV文件
    files = {
        'split.csv': pd.read_csv('split.csv'),
        'metadata.csv': pd.read_csv('metadata.csv'),
        'labels.csv': pd.read_csv('labels.csv'),
        'reports.csv': pd.read_csv('reports.csv')
    }

    # 1. 检查列名
    expected_columns = {
        'split.csv': ['subject_id', 'study_id', 'dicom_id', 'split'],
        'metadata.csv': ['subject_id', 'study_id', 'dicom_id', 'ViewPosition'],
        'labels.csv': ['study_id', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                       'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
                       'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
                       'Pneumothorax', 'Support Devices'],
        'reports.csv': ['study_id', 'findings', 'impression']
    }

    for file_name, df in files.items():
        expected = expected_columns[file_name]
        actual = list(df.columns)
        assert actual == expected, f"{file_name}列名不匹配: 期望{expected}, 实际{actual}"
        print(f"✅ {file_name} 列名正确")

    # 2. 检查ID一致性
    split_ids = set(files['split.csv']['study_id'])
    metadata_ids = set(files['metadata.csv']['study_id'])
    labels_ids = set(files['labels.csv']['study_id'])
    reports_ids = set(files['reports.csv']['study_id'])

    assert split_ids == metadata_ids == labels_ids == reports_ids, "study_id不一致"
    print("✅ 所有文件的study_id一致")

    # 3. 检查features.h5格式
    with h5py.File('features.h5', 'r') as f:
        # 检查路径格式
        expected_paths = ['p1/s1/d1', 'p2/s2/d2', 'p3/s3/d3']
        for path in expected_paths:
            assert f'img_embed/{path}' in f, f"缺少特征路径: img_embed/{path}"
            assert f'img_proj/{path}' in f, f"缺少投影路径: img_proj/{path}"
        print("✅ features.h5 路径格式正确")

    print("🎉 数据集验证通过！")


if __name__ == "__main__":
    validate_dataset()