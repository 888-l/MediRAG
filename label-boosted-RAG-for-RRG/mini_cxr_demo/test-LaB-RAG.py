import pandas as pd
import h5py
import numpy as np


def final_validation():
    print("🔍 最终数据验证...")

    # 1. 检查CSV文件
    try:
        split_df = pd.read_csv('split.csv')
        metadata_df = pd.read_csv('metadata.csv')
        labels_df = pd.read_csv('labels.csv')
        reports_df = pd.read_csv('reports.csv')
        print("✅ 所有CSV文件可正常读取")
    except Exception as e:
        print(f"❌ CSV文件读取失败: {e}")
        return

    # 2. 检查features.h5
    try:
        with h5py.File('features.h5', 'r') as f:
            # 检查所有样本的特征是否存在
            samples = [
                ('p1', 's1', 'd1'),
                ('p2', 's2', 'd2'),
                ('p3', 's3', 'd3')
            ]

            for subject, study, dicom in samples:
                path = f"{subject}/{study}/{dicom}"
                assert f'img_embed/{path}' in f, f"缺少img_embed: {path}"
                assert f'img_proj/{path}' in f, f"缺少img_proj: {path}"

            print("✅ features.h5 结构正确")

    except Exception as e:
        print(f"❌ features.h5 检查失败: {e}")
        return

    # 3. 检查数据一致性
    split_studies = set(split_df['study_id'])
    metadata_studies = set(metadata_df['study_id'])
    labels_studies = set(labels_df['study_id'])
    reports_studies = set(reports_df['study_id'])

    if split_studies == metadata_studies == labels_studies == reports_studies:
        print("✅ 所有文件的study_id一致")
    else:
        print("❌ study_id不一致")
        return

    print("🎉 数据集完全准备就绪！可以运行LaB-RAG测试了！")


if __name__ == "__main__":
    final_validation()