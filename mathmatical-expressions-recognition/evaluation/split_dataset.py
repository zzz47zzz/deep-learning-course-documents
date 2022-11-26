from genericpath import isdir, isfile
import os
import numpy
import random
import shutil

image_input_dir = './images/output_images'
label_input_dir = './images/output_annotations'
datasets_dir = './images/datasets'

train_path = os.path.join(datasets_dir,'train')
dev_path = os.path.join(datasets_dir,'dev')
test_path = os.path.join(datasets_dir,'test')

train_ids_path = os.path.join(datasets_dir,'train_ids.txt')
dev_ids_path = os.path.join(datasets_dir,'dev_ids.txt')
test_ids_path = os.path.join(datasets_dir,'test_ids.txt')

train_img_path = os.path.join(train_path,'images')
train_label_path = os.path.join(train_path,'labels')
dev_img_path = os.path.join(dev_path,'images')
dev_label_path = os.path.join(dev_path,'labels')
test_img_path = os.path.join(test_path,'images')
test_label_path = os.path.join(test_path,'labels')

if not os.path.isdir(datasets_dir):
    os.makedirs(datasets_dir)
if not os.path.isdir(train_img_path):
    os.makedirs(train_img_path)
if not os.path.isdir(train_label_path):
    os.makedirs(train_label_path)
if not os.path.isdir(dev_img_path):
    os.makedirs(dev_img_path)
if not os.path.isdir(dev_label_path):
    os.makedirs(dev_label_path)
if not os.path.isdir(test_img_path):
    os.makedirs(test_img_path)
if not os.path.isdir(test_label_path):
    os.makedirs(test_label_path)

img_lst = [n.split('.')[0] for n in os.listdir(image_input_dir)]
label_lst = [n.split('.')[0] for n in os.listdir(label_input_dir)]

assert len(label_lst)==len(img_lst), str(len(label_lst))+' '+str(len(img_lst))
assert set(img_lst)==set(label_lst)
assert len(set(img_lst))==len(img_lst)

# ==================================== stage 1: shuffle =====================================
sample_id_lst = img_lst
num_all_samples = len(sample_id_lst)
if os.path.isfile(train_ids_path) and os.path.isfile(dev_ids_path) and os.path.isfile(test_ids_path):
    train_ids = []
    dev_ids = []
    test_ids = []
    with open(train_ids_path,'r') as f:
        for line in f.readlines():
            train_ids.append(int(line))
    with open(dev_ids_path,'r') as f:
        for line in f.readlines():
            dev_ids.append(int(line))
    with open(test_ids_path,'r') as f:
        for line in f.readlines():
            test_ids.append(int(line))
else:
    random.shuffle(sample_id_lst)

    train_ids = sample_id_lst[:int(num_all_samples*(5/7))]
    dev_ids = sample_id_lst[int(num_all_samples*(5/7)):int(num_all_samples*(6/7))]
    test_ids = sample_id_lst[int(num_all_samples*(6/7)):]

    with open(train_ids_path,'w') as f:
        for one_id in train_ids:
            f.write(one_id+'\n')
    with open(dev_ids_path,'w') as f:
        for one_id in dev_ids:
            f.write(one_id+'\n')
    with open(test_ids_path,'w') as f:
        for one_id in test_ids:
            f.write(one_id+'\n')
cnt = 0
for one_id in train_ids:
    # for images
    suffix='.png'
    src_path = os.path.join(image_input_dir, str(one_id)+'.png')
    if not os.path.isfile(src_path):
        suffix='.jpg'
        src_path = os.path.join(image_input_dir, str(one_id)+'.jpg')
    shutil.copy(src_path, os.path.join(train_img_path,str(one_id)+suffix))
    # for labels
    src_path = os.path.join(label_input_dir, str(one_id)+'.txt')
    shutil.copy(src_path, os.path.join(train_label_path, str(one_id)+'.txt'))
    cnt+=1
    print(cnt)
for one_id in dev_ids:
    # for images
    suffix='.png'
    src_path = os.path.join(image_input_dir, str(one_id)+'.png')
    if not os.path.isfile(src_path):
        suffix='.jpg'
        src_path = os.path.join(image_input_dir, str(one_id)+'.jpg')
    shutil.copy(src_path, os.path.join(dev_img_path,str(one_id)+suffix))
    # for labels
    src_path = os.path.join(label_input_dir, str(one_id)+'.txt')
    shutil.copy(src_path, os.path.join(dev_label_path, str(one_id)+'.txt'))
    cnt+=1
    print(cnt)
for one_id in test_ids:
    # for images
    suffix='.png'
    src_path = os.path.join(image_input_dir, str(one_id)+'.png')
    if not os.path.isfile(src_path):
        suffix='.jpg'
        src_path = os.path.join(image_input_dir, str(one_id)+'.jpg')
    shutil.copy(src_path, os.path.join(test_img_path,str(one_id)+suffix))
    # for labels
    src_path = os.path.join(label_input_dir, str(one_id)+'.txt')
    shutil.copy(src_path, os.path.join(test_label_path, str(one_id)+'.txt'))
    cnt+=1
    print(cnt)
