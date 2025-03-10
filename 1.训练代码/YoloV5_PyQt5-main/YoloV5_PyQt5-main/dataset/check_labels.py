"""
check数据集中的标签文件格式是否都是正确
正确格式：<class_id> <x_center> <y_center> <width> <height>
说明：可以先运行 xml2txt.py 文件，生成 yolo 格式的标签文件，然后再运行此文件进行 check。
"""
import os

label_dir = "D:/Users/JokerWong/Desktop/20003170226张梓萱代码/1.训练代码/YoloV5_PyQt5-main/YoloV5_PyQt5-main/dataset/labels" \
            "/train"  # 标签文件目录
for label_file in os.listdir(label_dir):
    with open(os.path.join(label_dir, label_file), 'r') as f:
        lines = f.readlines()
        for line in lines:
            values = line.strip().split()
            if len(values) != 5:
                print(f"Error in {label_file}: {line}")
