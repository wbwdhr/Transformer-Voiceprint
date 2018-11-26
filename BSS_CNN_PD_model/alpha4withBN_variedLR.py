# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 16:59:19 2018 Updated in Jul 17
CNN-说话人识别软件 - main with Batch Normalization based on Tensorflow
先使用2_shuffledlistmaker.py建立数据库
@author: Yue
"""
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import datetime

starttime = datetime.datetime.now() # 计时器
tf.set_random_seed(1)
np.random.seed(1)

'训练设置'
# =============================================================================
EPOCH = 100
IMAGE_MUMBER = 490
BATCH_SIZE = 14          #IMAGE_MUMBER / BATCH_SIZE 需要被整除      
LR = tf.Variable(0.001, dtype=tf.float32) #可变LR
#LR = 0.001 #固定LR
DROPOUT_RATE = 0.5
if IMAGE_MUMBER % BATCH_SIZE != 0:
    print("IMAGE_MUMBER should be divisible by BATCH_SIZE, RUNNING OUT");input()
# =============================================================================
TRAINING_IMAGE_PATH = "data/training_filepath.txt"   #训练图片路径集
TESTING_IMAGE_PATH = "data/testing_filepath.txt"     #测试图片路径集
TRAINING_LABEL_PATH = "data/training_filelable.txt"  #训练图片标签
TESTING_LABEL_PATH = "data/testing_filelable.txt"    #测试图片标签
TRAINING_CATALOG_PATH = "data/training_catalog.txt"  #训练图片索引
TESTING_CATALOG_PATH = "data/testing_catalog.txt"    #测试图片索引
# =============================================================================
IMAGE_HEIGHT = 720  #!!!!可被4整除
IMAGE_WIDTH = 540   #!!!!可被4整除
CHAR_SET_LEN = 4 #output layer 输出胞元数量 （数据有多少种，即噪声种类）
# =============================================================================
'相关函数'
tf_x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH]) / 255
image = tf.reshape(tf_x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
tf_y = tf.placeholder(tf.int32, [None, CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32) # 防止过拟合
tf_is_train = tf.placeholder(tf.bool, None) # BN用于training or testing
# =============================================================================
# convert to gray灰度转换
def convert2gray(img):#灰度转换
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img
# =============================================================================
# labels to numbers & reverse                数字标签生成,分步代码见label2num.py
def label2num(label_path, catalog):#正向操作
    label_string = open(label_path, 'r', encoding="utf-8").read()
    label_string = ''.join(label_string) 
    labels = label_string.split('\n')#本句也可用此替代labels = label_string.splitlines() 
    catalog_string = open(catalog, 'r', encoding="utf-8").read()
    catalog_string = ''.join(catalog_string) 
    catalogs = catalog_string.split('\n')
    N = len(catalogs)
    num = len(labels)
    numbers = np.zeros([num,1],dtype = int)  
    for i in range(num):
        for k in range(N):
            m = k
            if catalogs[k] == labels[i]:
                numbers[i] = m
                break
    return numbers,num,catalogs #返回数字标签、标签个数、寻址字典

def num2label(numbers,catalogs):#逆向操作
    back = []
    for i in range(len(numbers)):
        m = (numbers[i])[0]
        if ord(m) <=57 : 
            k = int(m)   #将数组中的0-9保留
        else : 
            k = ord(m) - 55        #将数组中的A-Z 替换为10-35
        back.append(catalogs[k])
    return back
# =============================================================================
# numbers to vectors & reverse                 转换为向量,分步代码见label2num.py
def num2vec(numbers, char_set_len): # 正向操作
    vector = np.zeros(1 * char_set_len)
    for i, c in enumerate(numbers): # 遍历数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        idx = i * char_set_len + c  # char2pos(c)
        vector[idx] = 1
    return vector

def vec2num(vec): # 逆向操作
    char_pos = vec.nonzero()[0]
    numbers = []
    for i, c in enumerate(char_pos):
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        numbers.append(chr(char_code))
    return "".join(numbers)

'''CNN Block'''
# batch normalization for inputs 
image = tf.layers.batch_normalization(image, training = tf_is_train)
# 1st convolutional layer -- conv1 shape (28,28,1) -> (28,28,16) -> (14,14,16)
conv1 = tf.layers.conv2d(inputs=image, filters=16, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2,)
                          # BN for conv1
pool1 = tf.layers.batch_normalization(pool1, momentum = 0.4, training = tf_is_train)

# 2nd convolutional layer -- conv2 shape (14,14,16) -> (14,14,32) -> (7,7,32)
conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
# 3 dim to 1 row -- (7,7,32) -> (7*7*32)
flat = tf.reshape(pool2, [-1, 180*135*32]) 
# 1st fully connected layer -- 32 -> 128
fc1 = tf.layers.dense(flat, 128)
# 2st fully connected layer -- 128 -> 1024
fc2 = tf.layers.dense(fc1, 1024)
fc2 = tf.layers.dropout(fc2, keep_prob)
# 3rd fully connected layer / output layer -- 1024 -> 10
fc3 = tf.layers.dense(fc2, CHAR_SET_LEN)
output = fc3 #output = tf.layer.dense(flat, CHAR_SET_LEN) 

# compute cost for training data
loss = tf.losses.softmax_cross_entropy(onehot_labels = tf_y, logits = output)

# the moving_mean and moving_variance need to be updated, 
# pass the update_ops with control_dependencies to the train_op
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)

# compute accuracy for testing data
accuracy = tf.metrics.accuracy(labels = tf.argmax(tf_y,axis=1), predictions = tf.argmax(output,axis=1),)
accuracy = accuracy[1] # cause tf.metrics.accuracy will return (acc, update_op) and create 2 local var

# initializer
sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) #local var init is for accuracy_op
sess.run(init_op)

# visualization
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('\nPlease install sklearn for layer visualization\n')
def plot_with_labels(lowDWeights, labels):
    plt.cla(); # clear axis
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9));  # s为labels值，使用rainbow配色显示
        plt.text(x, y, s, backgroundcolor=c, fontsize=9) # 绘制输出点
    plt.xlim(X.min(), X.max());  # 设置x轴刻度范围
    plt.ylim(Y.min(), Y.max());  # 设置y轴刻度范围
    plt.title('Visualize last layer');  # 设置标题
    plt.show(); 
    plt.pause(0.01)  #显示秒数（若训练过快则会经过0.01s即覆盖）
    
def get_image_path(image_path): # 载入图片文件路径
    path_string = open(image_path, 'r', encoding="utf-8").read()
    path_string = ''.join(path_string) 
    image = path_string.split("\n")
    return image
TRAINING_IMG = get_image_path(TRAINING_IMAGE_PATH) 
TESTING_IMG = get_image_path(TESTING_IMAGE_PATH)

def get_next_batch(batch_size,step,images,image_labels_paths,catalog,char_set_len,mode): # batch分批设置 
    numbers,num,catalogs = label2num(image_labels_paths, catalog) #载入训练集索引文件
    batch_x = np.zeros([batch_size,IMAGE_HEIGHT * IMAGE_WIDTH]) #!![batch_size , 乘的关系]
    batch_y = np.zeros([batch_size,char_set_len])        
    for i in range(batch_size):
        if mode == 0: image_num = step * batch_size + i # 使得每个外循环都能加载下一批训练集
        else: image_num = i # 使得每次选用测试集时都为同一批
        label = numbers[image_num]
        image_path = images[image_num]
        load_image = Image.open(image_path)
        loaded_image = np.array(load_image) 
        
        image = convert2gray(loaded_image)
        batch_x[i,:] = image.flatten()/255
        batch_y[i,:] = num2vec(label, char_set_len)
    return batch_x, batch_y

plt.ion()
for epoch in range(EPOCH):
    if epoch < 50: sess.run(tf.assign(LR, 0.01* (0.92 ** epoch)))          #可变LR
    else: sess.run(tf.assign(LR, 0.001 * (0.96 ** epoch)))  #可变LR
#    sess.run(tf.assign(LR, 0.001 * (0.965 ** epoch))) #可变LR
    learning_rate = sess.run(LR)
    for step in range(int(IMAGE_MUMBER / BATCH_SIZE)):
        b_x, b_y =  get_next_batch(BATCH_SIZE, step, TRAINING_IMG, TRAINING_LABEL_PATH, TRAINING_CATALOG_PATH, CHAR_SET_LEN, 0)
        _, loss_ = sess.run([train_op,loss], feed_dict = {tf_x: b_x, tf_y: b_y, keep_prob: DROPOUT_RATE,  tf_is_train: True})
        if (step * BATCH_SIZE) % (BATCH_SIZE * 2) == 0:
            t_x, t_y = get_next_batch(BATCH_SIZE, step, TESTING_IMG, TESTING_LABEL_PATH, TESTING_CATALOG_PATH, CHAR_SET_LEN, 1)
            accuracy_, flat_representation = sess.run([accuracy, fc2], {tf_x: t_x, tf_y: t_y, keep_prob:1, tf_is_train: False})
            if step % 5 == 0:                
                print('Epoch:', epoch, '| Step:', step * BATCH_SIZE, '| train loss: %.5f' %loss_, '| test accuracy: %.3f' % accuracy_,'| learning rate:', str(learning_rate))
#                if step % 16 == 0: # Visualization of trained flatten layer (T-SNE) # Can be taken down
#                    if HAS_SK: 
#                        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000); plot_only = 1000
#                        low_dim_embs = tsne.fit_transform(flat_representation[:plot_only, :])
#                        labels = np.argmax(t_y, axis=1)[:plot_only]; plot_with_labels(low_dim_embs, labels)
plt.ioff()

endtime = datetime.datetime.now()
print("Time cost:", endtime - starttime)   

 # print 10 predictions from test data 展示测试集中的10个输入的预测值及实际值
test_output = sess.run(output, {tf_x: t_x[:15], keep_prob:1, tf_is_train: False})
pred_y = np.argmax(test_output, 1)
print(pred_y, 'prediction number')
print(np.argmax(t_y[:15], 1), 'real number')