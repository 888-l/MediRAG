import json
import nltk


def load_vocab(vocab_path):
    """加载词汇表JSON文件"""
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    return vocab_data['word2idx']


def convert_text_to_indices(text_file, vocab_file, output_file):
    """将文本文件中的单词转换为编码"""
    # 加载词汇表
    word2idx = load_vocab(vocab_file)

    # 读取并处理文本文件
    with open(text_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            # 分词并转换为小写
            tokens = nltk.tokenize.word_tokenize(line.lower())

            # 转换为编码
            indices = []
            for token in tokens:
                if token in word2idx:
                    indices.append(str(word2idx[token]))
                else:
                    # 使用<unk>的编码处理未知词汇
                    indices.append(str(word2idx.get('<unk>', '0')))

            # 用逗号连接编码并写入输出文件
            f_out.write(','.join(indices) + '\n')

            # 可选：显示进度
            if line_num % 100 == 0:
                print(f"已处理 {line_num} 行")


if __name__ == '__main__':
    # 配置参数
    test_file = 'train.txt'  # 输入文件
    vocab_file = 'vocab.json'  # 词汇表JSON文件
    output_file = 'train_encoded.txt'  # 输出文件

    # 执行转换
    convert_text_to_indices(test_file, vocab_file, output_file)
    print(f"转换完成！输出文件: {output_file}")