import tensorflow as tf
import os
import cv2
import numpy as np


def h_write(dim_2_array, path, micro_name):
    '''
    dim_2_array: 二维的矩阵
    path: 将头文件写入的路径
    micro_name: 头文件的记录的宏名称
    '''
    arr = np.array(dim_2_array)
    row, col = arr.shape
    content = ''
    for i, w in enumerate(dim_2_array):
        v = ','.join(str(c) for c in w)
        if i == row - 1:
            content = content + "$" + v + '*'
        else:
            content = content + "$" + v + '*, '
    content = '$' + content + '*'

    s = '#pragma once\n#define {} {}'.format(
        micro_name.upper(),
        content
    )

    s = s.replace('$', '{').replace('*', '}')
    with open(path, 'w') as f:
        f.write(s)


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data('/home/dongzw/Desktop/fpga/hls_lstm_v1/input/mnist/mnist.npz')

for i in range(10):
    name = os.path.join('./input/mnist/train', str(i) + '_' + str(y_train[i]) + '.jpg')
    img = x_train[i]
    cv2.imwrite(name, img)
    name = os.path.join('./input/mnist/train', str(i) + '_' + str(y_train[i]) + '.h')
    h_write(img, name, 'IMG_{}_{}'.format(i, y_train[i]))