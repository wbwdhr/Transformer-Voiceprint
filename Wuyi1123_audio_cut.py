"""
武义噪声数据截取
data里的数据嵌套方式如图所示
"""
import os
import numpy as np
import librosa
import math

filepath = []           # 建立一个空列表

InputType = ".wav"      # 输入文件类型
OutputType = ".wav"     # 输出图片类型
SR = 96000              # 采样频率 🌟 修改这里；本次数据均为96kHz

FORMAT = OutputType.lstrip('.')


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


def cut(wave, duration=30, mode='median'):  # 持续时间默认为30s, 取样方式默认为median中间值
    if mode == 'median':
        y, sr = librosa.load(wave, sr=SR)  # 读取一段完整数据
        total_t = librosa.get_duration(y, sr)
        half_t = math.floor(total_t/2)
        st = half_t - duration/2
        y_cut, sr = librosa.load(wave, offset=st, duration=duration, sr=SR)

    wave_label = wave.split('/')[-1]             # split 以/为path的切片符，[-1]向右，得到文件名与扩展名
    wave_name = wave_label.rstrip(InputType)     # 去除扩展名，得到文件名
    wave_source = wave.split('/')[-2]             # split 以/为path的切片符，[1]向左，得到上层文件夹的名
    if not os.path.exists('./cut/'):
        os.makedirs('./cut/')  # 如果以该数据文件名命名的文件夹不存在，则创建
    output_path = './cut/' + wave_source + '_' + wave_name + '.wav'
    librosa.output.write_wav(output_path, y_cut, sr)


FILE_PATH = get_all_files('./data/')    # 数据存放目录 🌟 修改这里
N = len(FILE_PATH)
for n in range(N):
    input_data = FILE_PATH[n]
    cut(input_data, duration=30, mode='median')
print('Done!')
