import os
from PIL import Image

path = "D:/Users/JokerWong/Desktop/Combined Dataset/train/JPEGImages/"
file_list = os.listdir(path)
print(file_list)
for file in file_list:
    img = Image.open(path + file)
    print(img.size)
    ratio = img.width / img.height
    img_resize = img.resize((640, 640), Image.BILINEAR)
    img_resize.save("D:/Users/JokerWong/Desktop/Combined Dataset/train/111/"+file)
