import os
import random
import shutil

# shuffle

input_dir = '../data/math_210421/formula_labels_210421/'
output_dir = '../data/math_210421/'
output_file = '../data/math_210421/im2latex_formulas.norm.lst'
image_input_dir = '../data/math_210421/formula_images_210421/'
# image_output_dir = '../data/math_210421/formula_images/'
image_name_list = os.listdir(image_input_dir)

label_name_list = os.listdir(input_dir)
random.shuffle(label_name_list)

total_num = len(label_name_list)
print(total_num)
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


with open(output_file, 'w', encoding='utf-8') as f0:


    index = 0
    with open(output_dir + 'im2latex_train_filter.lst', 'w', encoding='utf-8') as f1:
        for i in range(train_num):
            print(index, end='\r')
            train_label_name = train_list[i]
            image_name = train_label_name[:-4] + '.png'
            if image_name in image_name_list:
                f1.write(image_name + ' ' + str(index) + '\n')
                # shutil.copy(image_input_dir + image_name, image_output_dir + 'images_train/' + str(i) + '.png')
                with open(input_dir + train_label_name, 'r', encoding='utf-8') as f2:
                    line = f2.read()
                    f0.write(line + '\n')
                index += 1

    with open(output_dir + 'im2latex_validate_filter.lst', 'w', encoding='utf-8') as f1:
        for i in range(val_num):
            print(index, end='\r')
            val_label_name = val_list[i]
            image_name = val_label_name[:-4] + '.png'
            if image_name in image_name_list:
                f1.write(image_name + ' ' + str(index) + '\n')
                # shutil.copy(image_input_dir + image_name, image_output_dir + 'images_val/' + str(i) + '.png')
                with open(input_dir + val_label_name, 'r', encoding='utf-8') as f2:
                    line = f2.read()
                    f0.write(line + '\n')
                index += 1

    with open(output_dir + 'im2latex_test_filter.lst', 'w', encoding='utf-8') as f1:
        for i in range(test_num):
            print(index, end='\r')
            test_label_name = test_list[i]
            image_name = test_label_name[:-4] + '.png'
            if image_name in image_name_list:
                f1.write(image_name + ' ' + str(index) + '\n')
                # shutil.copy(image_input_dir + image_name, image_output_dir + 'images_test/' + str(i) + '.png')
                with open(input_dir + test_label_name, 'r', encoding='utf-8') as f2:
                    line = f2.read()
                    f0.write(line + '\n')
                index += 1
    


# shuffle end