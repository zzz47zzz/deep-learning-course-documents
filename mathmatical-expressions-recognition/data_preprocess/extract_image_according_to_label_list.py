import os
import shutil

label_dir = './data/math_210421/formula_labels_210421_no_chinese/'
image_dir = './data/math_210421/formula_images/'
output_dir ='./data/math_210421/formula_images_210421/'

label_name_list = os.listdir(label_dir)

for i in range(len(label_name_list)):
    label_name_list[i] = label_name_list[i][:-4]

# print(label_list)

image_name_list = os.listdir(image_dir)

for image_name in image_name_list:
    if image_name[:-4] in label_name_list:
        print(image_name)
        shutil.copy(image_dir + image_name, output_dir + image_name)