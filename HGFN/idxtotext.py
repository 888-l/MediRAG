import json
import argparse


def load_vocab(vocab_path):
    """加载词汇表JSON文件"""
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    return vocab_data['idx2word']


def convert_indices_to_text(encoded_file, vocab_file, output_file):
    """将编码文件转换回文本"""
    # 加载词汇表
    idx2word = load_vocab(vocab_file)

    # 读取并处理编码文件
    with open(encoded_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                f_out.write('\n')
                continue

            # 分割编码
            indices = line.split(',')

            # 转换为单词
            words = []
            for idx_str in indices:
                idx = int(idx_str)
                if str(idx) in idx2word:
                    word = idx2word[str(idx)]
                    # 跳过特殊标记（可选，根据需要调整）
                    if word not in ['<pad>', '<start>', '<end>']:
                        words.append(word)
                else:
                    words.append('<unk>')

            # 组合成句子并写入输出文件
            sentence = ' '.join(words)
            f_out.write(sentence + '\n')

            # 可选：显示进度
            if line_num % 100 == 0:
                print(f"已处理 {line_num} 行")


def convert_indices_to_text_with_special_tokens(encoded_file, vocab_file, output_file):
    """将编码文件转换回文本（保留所有特殊标记）"""
    # 加载词汇表
    idx2word = load_vocab(vocab_file)

    with open(encoded_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                f_out.write('\n')
                continue

            indices = line.split(',')
            words = []
            for idx_str in indices:
                idx = int(idx_str)
                if str(idx) in idx2word:
                    words.append(idx2word[str(idx)])
                else:
                    words.append('<unk>')

            sentence = ' '.join(words)
            f_out.write(sentence + '\n')

            if line_num % 100 == 0:
                print(f"已处理 {line_num} 行")


if __name__ == '__main__':
    # 配置参数
    encoded_file = 'test.txt'  # 输入的编码文件
    vocab_file = 'vocab.json'  # 词汇表JSON文件
    output_file = 'result.txt'  # 输出的文本文件

    # 选择转换方式：
    # 1. 过滤特殊标记的版本
    convert_indices_to_text(encoded_file, vocab_file, output_file)

    # 2. 保留所有特殊标记的版本
    # convert_indices_to_text_with_special_tokens(encoded_file, vocab_file, output_file)

    print(f"反向转换完成！输出文件: {output_file}")