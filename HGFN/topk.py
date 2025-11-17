import evaluation
import os

model_path = "./runs/runX/checkpoint/NTN_relation/model_best.pth.tar"
# model_path = "./runs/runX/checkpoint/NTN_relation/model_best.pth.tar"
data_path = "./data/"


def extract_texts_by_index(index_list, input_file, output_file):
    """
    根据索引列表从输入文件中提取对应的文本行
    
    Args:
        index_list: 索引列表（从0开始）
        input_file: 输入文本文件路径
        output_file: 输出文件路径
    """
    # 读取输入文件的所有行
    with open(input_file, 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f.readlines()]
    
    print(f"输入文件共有 {len(all_lines)} 行文本")
    print(f"需要提取 {len(index_list)} 个索引对应的文本")
    
    # 提取对应的文本行
    extracted_texts = []
    valid_count = 0
    invalid_count = 0
    
    for i, idx in enumerate(index_list):
        if 0 <= idx < len(all_lines):
            extracted_texts.append(all_lines[idx])
            valid_count += 1
        else:
            print(f"警告: 索引 {idx} 超出范围 (0-{len(all_lines)-1})")
            extracted_texts.append("")  # 无效索引输出空行
            invalid_count += 1
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in extracted_texts:
            f.write(text + '\n')
    
    print(f"成功提取 {valid_count} 行文本")
    if invalid_count > 0:
        print(f"无效索引: {invalid_count} 个")
    print(f"结果已保存到: {output_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='根据索引提取文本')
    parser.add_argument('--input_file', default='./data/dataset/graph/test/test.txt', 
                       help='输入文本文件路径，默认: test.txt')
    parser.add_argument('--output_file', default='output.txt', 
                       help='输出文件路径，默认: test.txt')
    
    args = parser.parse_args()
    top=evaluation.evaltop(model_path, data_path=data_path, split="test")
    extract_texts_by_index(top, args.input_file, args.output_file)

if __name__ == '__main__':
    main()
