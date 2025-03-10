import os
import random
import shutil
"""
划分数据集---->训练集和验证集
图片数据集格式为.jpg
标签数据集格式为.txt
train_ratio---->划分比例
"""
def split_dataset(dataset_path, train_ratio=0.9):
    image_folder = os.path.join(dataset_path, 'JPEGImages-640')  # 图片数据集的文件夹
    label_folder = os.path.join(dataset_path, 'Labels')  # 标签数据集的文件夹

    images = os.listdir(image_folder)
    random.shuffle(images)

    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]
    for dataset, image_list in [('train', train_images), ('val', val_images)]:
        if dataset == 'train':
            dataset_image_folder = os.path.join(dataset_path, dataset, 'images')
            print("训练集images文件夹已创建完成，路径为：", dataset_image_folder)
            dataset_label_folder = os.path.join(dataset_path, dataset, 'labels')
            print("训练集labels文件夹已创建完成，路径为：", dataset_label_folder)
        else:
            dataset_image_folder = os.path.join(dataset_path, dataset, 'images')
            print("验证集images文件夹已创建完成，路径为：", dataset_image_folder)
            dataset_label_folder = os.path.join(dataset_path, dataset, 'labels')
            print("验证集labels文件夹已创建完成，路径为：", dataset_label_folder)

        os.makedirs(dataset_image_folder, exist_ok=True)  # 创建image文件夹
        os.makedirs(dataset_label_folder, exist_ok=True)  # 创建label文件夹

        for image_file in image_list:
            image_path = os.path.join(image_folder, image_file)
            label_file = image_file.replace('.jpg', '.txt')
            label_path = os.path.join(label_folder, label_file)

            shutil.copy(image_path, os.path.join(dataset_image_folder, image_file))
            shutil.copy(label_path, os.path.join(dataset_label_folder, label_file))


# 指定路径
dataset_path = 'D:/Users/JokerWong/Desktop/Combined Dataset/train'
split_dataset(dataset_path)
