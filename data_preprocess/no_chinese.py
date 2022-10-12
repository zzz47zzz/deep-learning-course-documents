import os
import shutil

input_label_dir = './data/math_210421/formula_labels_210421_no_error_mathpix/'
output_label_dir = './data/math_210421/formula_labels_210421_no_chinese/'

label_name_list = os.listdir(input_label_dir)

with open('./data_preprocess/vocab.txt', 'r', encoding='utf-8') as f:
    vocab = f.read().split()

max_token_len = 0
for v in vocab:
    if len(v) > max_token_len:
        # print(len(v))
        max_token_len = len(v)
# print(max_token_len)

def FMM_func(user_dict, sentence):
    """
    正向最大匹配（FMM）
    :param user_dict: 词典
    :param sentence: 句子
    """
    # 词典中最长词长度
    max_len = max([len(item) for item in user_dict])
    start = 0
    token_list = []
    while start != len(sentence):
        index = start+max_len
        if index>len(sentence):
            index = len(sentence)
        for i in range(max_len):
            if (sentence[start:index] in user_dict) or (len(sentence[start:index])==1):
                token_list.append(sentence[start:index])
                # print(sentence[start:index], end='/')
                start = index
                break
            index += -1
    return token_list

chinese_token_list=[]

index = 1
for label_name in label_name_list:
    print(index, ':')
    index += 1
    # print(label_name, ':',end='')
    label_file_name = input_label_dir + label_name
    with open(label_file_name, 'r', encoding='utf-8') as f1:
        content = f1.read()

    # print(content)

    token_list = FMM_func(vocab, content)
    token_list = [token_list[i] for i in range(len(token_list)) if token_list[i] != ' '] # 去除空格
    # print(token_list)

    new_content = ' '.join(token_list)

    # print(new_content)
    
    have_chinese = False

    for token in token_list:
        if token not in vocab and token not in ['', ' ']:
            # print(label_name, ':',end='')
            # print(token)
            chinese_token_list.append(token)
            have_chinese = True

    if have_chinese is not True:
        # shutil.copy(label_file_name, output_label_dir + label_name)
        with open(output_label_dir + label_name, 'w', encoding='utf-8') as f:
            f.write(new_content)
    else:
        with open('./data/math_210421/formula_labels_210421_chinese/' + label_name, 'w', encoding='utf-8') as f:
            f.write(new_content)

    # if have_chinese is True:
    #     print()

with open('./data/math_210421/chinese_token.txt', 'w', encoding='utf-8') as f:
    chinese_token_list = list(set(chinese_token_list))
    for chinese_token in chinese_token_list:
        f.write(chinese_token + '\n')