# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 15:17:50 2018
图像处理压缩模块
检测data/train/目录下的所有指定扩展名图像文件，并将其等比例压缩至新的图像文件，可指定输出扩展名
!!!在listmarker.py前运行此程序
@author: Yue
"""
import os
from PIL import Image

# =========================path and name assignment============================
training_image_path = "data/train/"   #训练data存放位置
testing_image_path = "data/test/"     #测试data存放位置
old_extension_name = '.png'  #旧扩展名
new_extension_name = '.png' #新扩展名
# =============================================================================

def resize_image(image_file):
    new_width = 480
    new_height = 360
    path = image_file
    im = Image.open(path)
    im.thumbnail((new_width, new_height))
#    im.transpose(Image.FLIP_TOP_BOTTOM)  #垂直反转（已在生成图片时，使用了y轴转置，因此此处不需要垂直反转）
    im.save(path.replace(old_extension_name, new_extension_name)) #('原扩展名','新扩展名')

def get_all_files(path):
    files = os.listdir(path)              #读取path下的所有文件
    for fi in files:
        fi_d = os.path.join(path,fi)      #将读取到的文件与path组合成完整路径
        if os.path.isdir(fi_d):           #若该路径指向子文件夹，则重复次函数，即进入子目录
            get_all_files(fi_d)
        else:
            if fi.endswith(old_extension_name):  #若指向.XXX文件：
                image_path = path + "/" + fi     #文件路径  --> path list
                resize_image(image_path)

if __name__ == '__main__':
    get_all_files(training_image_path)
    get_all_files(testing_image_path)
    print("All " + old_extension_name + " images has been resized into " + new_extension_name + " wtih new resolution" )

