import os
import random
import shutil

# shuffle

input_dir = '../data/210618/formulas_labels/'
output_dir = '../data/210705/formulas/'
image_input_dir = '../data/210618/formulas_images/'
image_output_dir = '../data/210705/images/'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
if not os.path.isdir(image_output_dir):
    os.mkdir(image_output_dir)
    os.mkdir(image_output_dir + 'images_train/')
    os.mkdir(image_output_dir + 'images_val/')
    os.mkdir(image_output_dir + 'images_test/')

label_name_list = os.listdir(input_dir)
random.shuffle(label_name_list)

# total_num = len(label_name_list)
# train_num = int(0.7 * total_num)
# val_num = int(0.1 * total_num)
# test_num = total_num - train_num - val_num

total_num = 76253
train_num = int(0.7 * total_num)
val_num = int(0.1 * total_num)
test_num = total_num - train_num - val_num

train_list = []
val_list = []
test_list = []

for i in range(total_num):
    if i < train_num:
        train_list.append(label_name_list[i])
    elif i < train_num + val_num:
        val_list.append(label_name_list[i])
    else:
        test_list.append(label_name_list[i])

with open(output_dir + 'train.formulas.norm.txt', 'w', encoding='utf-8') as f1:
    for i in range(train_num):
        train_label_name = train_list[i]
        image_name = train_label_name[:-4] + '.png'
        shutil.copy(image_input_dir + image_name, image_output_dir + 'images_train/' + str(i) + '.png')
        with open(input_dir + train_label_name, 'r', encoding='utf-8') as f2:
            line = f2.read()
            f1.write(line + '\n')

with open(output_dir + 'val.formulas.norm.txt', 'w', encoding='utf-8') as f1:
    for i in range(val_num):
        val_label_name = val_list[i]
        image_name = val_label_name[:-4] + '.png'
        shutil.copy(image_input_dir + image_name, image_output_dir + 'images_val/' + str(i) + '.png')
        with open(input_dir + val_label_name, 'r', encoding='utf-8') as f2:
            line = f2.read()
            f1.write(line + '\n')

with open(output_dir + 'test.formulas.norm.txt', 'w', encoding='utf-8') as f1:
    for i in range(test_num):
        test_label_name = test_list[i]
        image_name = test_label_name[:-4] + '.png'
        shutil.copy(image_input_dir + image_name, image_output_dir + 'images_test/' + str(i) + '.png')
        with open(input_dir + test_label_name, 'r', encoding='utf-8') as f2:
            line = f2.read()
            f1.write(line + '\n')

# shuffle end