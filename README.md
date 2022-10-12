# 手写公式识别项目

## 1. 步骤一：标注数据

## 2. 步骤二：数据预处理（data_preprocess文件夹）

### 2.1 项目文件功能
```
.
├── data_filter.py # 过滤多行和内容为error mathpix的标签
├── data_preprocess_for_im2latex.py    # 将数据整理成im2latex这个项目需要的格式
├── extract_image_according_to_label_list.py    # 根据有效标签提取对应图片（一般来说有效标签数小于图片数，这一步是在预处理阶段将两个文件夹对齐，当然你也可以在模型的data_loader阶段对齐，总之以标签文件为锚点，不要出现根据图片去找标签这个情况，因为可能找不到。）
├── no_chinese.py   # 这个文件非常重要，首先根据vocab（vocab关键词不完整，欢迎大家人工添加）进行分词，再过滤不在词表的标签文件
├── pad_image.py    # 做图片padding的
├── shuffle_and_build_dataset.py    # 针对LaTeX_OCR_PRO这个项目的格式预处理
├── vocab.txt
└── write_matching.py
```


### 2.2 预处理思路 
    1. Tokenization，根据词表进行分词，并根据词表初步过滤数据
    2. 过滤多行数据和error mathpix
    3. 对齐过滤后的数据
    4. 根据项目输入输出格式对数据进行最后的调整
    5. 根据神经网络模型的需要，看是否需要padding，padding到什么size

## 3. 步骤三：训练模型

## 4. 步骤四：评估和测试模型