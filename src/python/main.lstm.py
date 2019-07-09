'''
Date: 2019-7-7

model input shape: (?, 28, 28)
model weights: dtype=float32

layer lstm:
>0: shape=(28, 1024)      kernel
>1: shape=(256, 1024)     recurrent_kernel
>2: shape=(1024,)         bias

layer dense:
>0: shape=(256, 256)
>1: shape=(256,)

layer dense:
>0 shape=(256, 10)
>1 shape=(10)

60000/60000 [==============================] - 593s 10ms/sample - loss: 0.2759 - acc: 0.9133 - val_loss: 0.1096 - val_acc: 0.9662
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm (LSTM)                  (None, 256)               291840    
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense (Dense)                (None, 256)               65792     
_________________________________________________________________
softmax (Dense)              (None, 10)                2570      
=================================================================
Total params: 360,202
Trainable params: 360,202
Non-trainable params: 0
_________________________________________________________________
'''

import tensorflow as tf
import numpy as np
import os


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
    
    s += '\n#define {} {}'.format(micro_name.upper() + '_WIDTH', row)
    s += '\n#define {} {}'.format(micro_name.upper() + '_HEIGHT', col)
    s = s.replace('$', '{').replace('*', '}')
    with open(path, 'w') as f:
        f.write(s)


def get_model(load_path=None):
    if load_path is None:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(units=256, input_shape=(28, 28)))
        model.add(tf.keras.layers.Dropout(rate=0.2))
        model.add(tf.keras.layers.Dense(256, activation='relu', name='dense'))
        model.add(tf.keras.layers.Dense(10, activation='softmax', name='softmax'))

        model.compile(
                        loss='categorical_crossentropy',
                        optimizer='rmsprop',
                        metrics=['accuracy']
                        )
    else:
        model = tf.keras.models.load_model(load_path)
    return model


def save_weights_to_h(model):
    units = 256
    if not os.path.exists('./weight'):
        os.mkdir('./weight')
    for layer in model.layers:
        weights = layer.get_weights()
        names = [x.name.split('/')[-1].split(':')[0] for x in layer.weights]
        if layer.name == 'lstm':
            for weight, name in zip(weights, names):
                h_list = []
                _v = ['i', 'f', 'c', 'o']
                if name == 'kernel':
                    kernel_i = weight[:, :units]
                    kernel_f = weight[:, units: units*2]
                    kernel_c = weight[:, units*2: units*3]
                    kernel_o = weight[:, units*3:]
                    h_list.append(kernel_i)
                    h_list.append(kernel_f)
                    h_list.append(kernel_c)
                    h_list.append(kernel_o)
                elif name == 'recurrent_kernel':
                    recurrent_kernel_i = weight[:, :units]
                    recurrent_kernel_f = weight[:, units: units*2]
                    recurrent_kernel_c = weight[:, units*2: units*3]
                    recurrent_kernel_o = weight[:, units*3:]
                    h_list.append(recurrent_kernel_i)
                    h_list.append(recurrent_kernel_f)
                    h_list.append(recurrent_kernel_c)
                    h_list.append(recurrent_kernel_o)
                elif name == 'bias':
                    # bias_i = weight[:units]
                    # bias_f = weight[units: units*2]
                    # bias_c = weight[units*2: units*3]
                    # bias_o = weight[units*3:]
                    for i, v in enumerate(_v):
                        path = os.path.join('./weight', layer.name+ '_' + name + '_' + _v[i] + '.h')
                        s = '#define {}_{}_{} $'.format(layer.name.upper(), name.upper(), v.upper())
                        s += ','.join(str(i) for i in weight[i * units:(i+1) * units])
                        s += '*'
                        s = s.replace('$', '{').replace('*', '}')
                        with open(path, 'w') as f:
                            f.write(s)

                for i, h in enumerate(h_list):
                    path = os.path.join('./weight', layer.name+ '_' + name + '_' + _v[i] + '.h')
                    h_write(h, path, "W_{}_{}_{}".format(layer.name, name, _v[i]))


        else:
            for weight, name in zip(weights, names):
                path = os.path.join('./weight', layer.name +'_'+ name + '.h')
                if name == 'bias':
                    # 是bias
                    s = '#pragma once\n#define W_{}_BIAS ${}*'.format(
                        layer.name.upper(),
                        ','.join(str(c) for c in weight)
                    )
                    s = s.replace('$', '{').replace('*', '}')
                    with open(path, 'w') as f:
                        f.write(s)
                elif name == 'kernel':
                    h_write(weight, path, "W_{}_KERNEL".format(layer.name))


def train():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data('/home/dongzw/Desktop/fpga/hls_lstm/input/mnist/mnist.npz')

    x_train = x_train / 128.0 - 1
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    x_test = x_test / 128.0 - 1
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # model = get_model('./models/keras.h5')
    model = get_model()
    model.fit(x=x_train, y=y_train, batch_size=20, epochs=5, validation_data=(x_test, y_test))
    save_weights_to_h(model)

    model.summary()
    if not os.path.exists('./models'):
        os.mkdir('./models')
    model.save('./models/keras.h5', save_format='h5')
    model.save_weights('./models/lstm.weights.h5', save_format='h5')


if __name__ == '__main__':
    train()
