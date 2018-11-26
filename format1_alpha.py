"""
Format.1 数据格式化 wav/mat to png
2018.11.18 - Yue - MacOS PyCharm
采集到的样本并不总是可以直接拿过来使用的，需要对数据进行整理分类，打标签，称之为格式化。
本程序将采集到的数据信号进行时频谱图，将完整的一段信号转换为一幅完整的时频谱图。（Format.1）
然后将一张大的时频谱图切分成一张一张的Height*Weight大小的时频谱图。（见Format.2）
"""

from __future__ import print_function
from scipy.io import loadmat
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
filepath = []           # 建立一个空列表

InputType = ".wav"      # 输入文件类型
OutputType = ".png"     # 输出图片类型
SR = 96000              # 采样频率 🌟 修改这里

FORMAT = OutputType.lstrip('.')


def extractor(file, input_type=InputType):  # 数据提取

    if input_type == ".mat":
        mat = loadmat(file)  # 读取一段完整数据
        mat = mat['range']   # 根据mat的变量名称进行修改
        y = mat.reshape(len(mat))
        sr = SR
        y = y - y.mean()  # 增加区分度
    if input_type == ".wav":
        y, sr = librosa.load(file, sr=SR)    # 读取一段完整数据
        y = y - y.mean()    # 增加区分度
    # else:
    #     input(), print('Fuxk u')  # 跳出

    # 以下内容为前景背景分离
    s, phase = librosa.magphase(librosa.stft(y))

    s_filter = librosa.decompose.nn_filter(s, aggregate=np.median, metric='cosine',
                                           width=int(librosa.time_to_frames(5, sr=SR)))  # 5s / 25s
    # To avoid being biased by local continuity, we constrain similar frames to be separated by at least 2 seconds.
    # librosa.time_to_frames(5, sr=SR) 此处的5s应至少小于数据的原始时长。！！！
    s_filter = np.minimum(s, s_filter)

    margin_i, margin_v = 2, 10

    mask_i = librosa.util.softmask(s_filter, margin_i*(s-s_filter), power=2)
    mask_v = librosa.util.softmask(s-s_filter, margin_v*s_filter, power=2)

    s_foreground = mask_v * s
    s_background = mask_i * s

    yf = librosa.istft(s_foreground * phase)
    yb = librosa.istft(s_background * phase)
    return y, yf, yb


def auto_resolution(y, yf, yb):  # 自适应分辨率计算
    # 先对数据进行补齐操作
    length_y = len(y)
    length_yf = len(yf)
    length_yb = len(yb)
    max_length = max(length_y, length_yb, length_yf)

    def zero_fill(x, m_l= max_length):
        x_fixed = x
        if len(x) < m_l:
            deta = abs(len(x) - m_l)
            x_fixed = np.pad(x, (0, deta), 'constant', constant_values=0)
        return x_fixed
    y = zero_fill(y)
    yf = zero_fill(yf)
    yb = zero_fill(yb)
    w = max_length/80  # 输出的宽度（与数据的持续时间有关）
    h = 224            # 输出的高度（与数据的采样频率有关）
    if SR >= 96000:
        h = 448
    if SR == 44800:
        h = 448
    return y, yf, yb, w, h


def saver(data, name='_', width=224, height=224):  # 默认224*224
    width = width/100; height = height/100
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(width/3, height/3)  # dpi = 300, output = 224*224 pixels
    plt.specgram(data, NFFT=1024, Fs=SR, noverlap=512, cmap='jet')
    # 反转y轴，与MATLAB默认一致
    plt.gca().invert_yaxis()
    # 去掉白边，减少无用的padding
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    fig.savefig(name + OutputType, format=FORMAT, transparent=True, dpi=300, pad_inches=0)


def get_all_files(path):
    files = os.listdir(path)                # 读取path下的所有文件
    for fi in files:
        fi_d = os.path.join(path, fi)       # 将读取到的文件与path组合成完整路径
        if os.path.isdir(fi_d):             # 若该路径指向子文件夹，则重复次函数，即进入子目录
            get_all_files(fi_d)
        else:
            if fi.endswith(InputType):      # !!!! 若指向.wav文件：
                filepath.append(path + '/' + fi)
    return filepath


FILE_PATH = get_all_files('./cut/')                # 数据存放目录 🌟 修改这里
N = len(FILE_PATH)
for i in range(N):
    input_data = FILE_PATH[i]
    y, yf, yb = extractor(input_data)               # 数据提取
    y, yf, yb, w, h = auto_resolution(y, yf, yb)    # 自适应分辨率计算
    OutputName = input_data.rstrip(InputType)
    saver(y, OutputName, w, h)                      # 输出分辨率 🌟 修改这里
print('Done!')
