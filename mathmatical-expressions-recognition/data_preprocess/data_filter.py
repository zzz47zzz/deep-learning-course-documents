# Created: 210313 14:02
# Last edited: 210421 14:02 

import os
import shutil


# # 筛除多行的label.txt

# input_label_dir = './data/math_210421/formula_labels/'
# output_label_dir = './data/math_210421/formula_labels_210421/'

# label_name_list = os.listdir(input_label_dir)

# for label_name in label_name_list:
#     label_file_name = input_label_dir + label_name
#     with open(label_file_name, 'r', encoding='utf-8') as f1:
#         lines = f1.readlines()
#     # print(lines)
#     if len(lines) > 1:
#         # print(lines[1])
#         shutil.copy(label_file_name, './data/math_210421/mult-line_label/' + label_name)
#         continue
#     shutil.copy(label_file_name, output_label_dir + label_name)

# # 筛除多行的label.txt end

# 筛除error mathpix

input_label_dir = './data/math_210421/formula_labels_210421/'
output_label_dir = './data/math_210421/formula_labels_210421_no_error_mathpix/'

label_name_list = os.listdir(input_label_dir)

for label_name in label_name_list:
    print(label_name)
    label_file_name = input_label_dir + label_name
    with open(label_file_name, 'r', encoding='utf-8') as f1:
        content = f1.read()
    if 'error mathpix' in content:
        print(content)
        continue
    shutil.copy(label_file_name, output_label_dir + label_name)

# 筛除error mathpix end