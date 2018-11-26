# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 22:50:23 2018
数据列表读取、随机化和保存模块
（数据保存格式要求!!!例：data/train/XXX/aaa.png，XXX为噪声标签，aaa.png为XXX噪声的一个片段）
@author: Yue
"""
import os
import random

# =============================================================================
itemtype = ".png"  #数据库的类型
trnum = 490       #训练集数量
tenum = 55        #测试集数量
# =============================================================================

#训练集和测试集数据存放位置
training_data_path = "data/train/"
testing_data_path = "data/test/"
#list存放位置
list_path="data/"

#training_filelist = []
training_filepath = []
training_filelabel = []
#testing_filelist = []
testing_filepath = []
testing_filelabel = []
training_catalog = []
testing_catalog = []

def get_all_files(path,catalog,filepath,filelabel):
    files = os.listdir(path)             #读取path下的所有文件
    for fi in files:
        fi_d = os.path.join(path,fi)       #将读取到的文件与path组合成完整路径
        if os.path.isdir(fi_d):                #若该路径指向子文件夹，则重复次函数，即进入子目录
            catalog.append(fi)
            get_all_files(fi_d,catalog,filepath,filelabel)
        else:
            if fi.endswith(itemtype):                  #!!!! 若指向.png文件：
#                filelist.append(fi)                  #文件名    --> file list
                filepath.append(path + "/" + fi)     #文件路径  --> path list
                filelabel.append((path.split('/')[-1]))#文件夹名  -->label list
                #split 以/为path的切片符，.txt文件前的最后一个子目录（-1指倒数第一个）的文件夹名
                
def randsort(filepath,filelabel,num):
    randnum = random.randint(1,num)   #为了保证标签与路径文件的随机方式一致（否则无法对应），保存随机方式
    random.seed(randnum)
    random.shuffle(filepath)
    random.seed(randnum)
    random.shuffle(filelabel)
    

def write_list(output,_list):
    text = '\r'.join(_list)      #用回车连接list中的多个条目
    f = open(list_path + output,'w')
    f.write(text)
    f.close
    
    
if __name__ == '__main__':
#1  #获取所有的训练集数据位置进行标签
    get_all_files(training_data_path,training_catalog,training_filepath,training_filelabel)
    #获取所有的测试集数据位置进行标签
    get_all_files(testing_data_path,testing_catalog,testing_filepath,testing_filelabel)
    print("DATA LOADED")
    
#2  #对训练集的数据（文件路径与标签）进行统一随机排序
    randsort(training_filepath,training_filelabel,trnum)
    #对测试集的数据（文件路径与标签）进行统一随机排序（与训练集的随机方式不同）
    randsort(testing_filepath,testing_filelabel,tenum)
    print("DATA SHUFFLED")
    
#3  #将训练集标签保存
    write_list("training_catalog.txt",training_catalog)
    write_list("training_filepath.txt",training_filepath)
    write_list("training_filelable.txt",training_filelabel)
    #将测试集标签保存
    write_list("testing_catalog.txt",testing_catalog)
    write_list("testing_filepath.txt",testing_filepath)
    write_list("testing_filelable.txt",testing_filelabel)
    print("SAVE DONE")