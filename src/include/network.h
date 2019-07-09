#pragma once
#include "utils.h"
#include "lstm.h"
#include "dense.h"
#include "softmax.h"

// 权重的初始化文件 
#include "../../weight/dense_kernel.h"
#include "../../weight/dense_bias.h"
#include "../../weight/lstm_bias_i.h"
#include "../../weight/lstm_kernel_f.h"
#include "../../weight/lstm_recurrent_kernel_c.h"
#include "../../weight/lstm_recurrent_kernel_o.h"
#include "../../weight/lstm_bias_c.h"
#include "../../weight/lstm_bias_o.h"
#include "../../weight/lstm_kernel_i.h"
#include "../../weight/lstm_recurrent_kernel_f.h"
#include "../../weight/softmax_kernel.h"
#include "../../weight/softmax_bias.h"
#include "../../weight/lstm_bias_f.h"
#include "../../weight/lstm_kernel_c.h"
#include "../../weight/lstm_kernel_o.h"
#include "../../weight/lstm_recurrent_kernel_i.h"


class Network
{
public:
    Network();
    void forward();
 public:
    // 将输入设置为 28  28 的图片输入；
    data_t arr_input[28][28];
    Matrix input = Matrix((data_t *)arr_input, 28, 28);
    // 其输出为一个 10维向量；
    // data_t arr_output[1][10];
    // Matrix output = Matrix((data_t *)arr_output, 1, 10);  

    LSTM lstm = LSTM(&w_lstm);
    Dense dense = Dense(&dense_kernel, &dense_bias);
    Softmax softmax = Softmax(&softmax_kernel, &softmax_bias);
    Matrix output = softmax.output;

private:
    // lstm 参数初始化
    data_t arr_kernel_i[TIME_STEPS][NUM_UNITS] = W_LSTM_KERNEL_I;
    data_t arr_kernel_f[TIME_STEPS][NUM_UNITS] = W_LSTM_KERNEL_F;
    data_t arr_kernel_c[TIME_STEPS][NUM_UNITS] = W_LSTM_KERNEL_C;
    data_t arr_kernel_o[TIME_STEPS][NUM_UNITS] = W_LSTM_KERNEL_O;

    Matrix kernel_i = Matrix((data_t *)arr_kernel_i, TIME_STEPS, NUM_UNITS);
    Matrix kernel_f = Matrix((data_t *)arr_kernel_f, TIME_STEPS, NUM_UNITS);
    Matrix kernel_c = Matrix((data_t *)arr_kernel_c, TIME_STEPS, NUM_UNITS);
    Matrix kernel_o = Matrix((data_t *)arr_kernel_o, TIME_STEPS, NUM_UNITS);
    
    // 与h相乘
    data_t arr_recurrent_kernel_i[NUM_UNITS][NUM_UNITS] = W_LSTM_RECURRENT_KERNEL_I;
    data_t arr_recurrent_kernel_f[NUM_UNITS][NUM_UNITS] = W_LSTM_RECURRENT_KERNEL_F;
    data_t arr_recurrent_kernel_c[NUM_UNITS][NUM_UNITS] = W_LSTM_RECURRENT_KERNEL_C;
    data_t arr_recurrent_kernel_o[NUM_UNITS][NUM_UNITS] = W_LSTM_RECURRENT_KERNEL_O;
    Matrix recurrent_kernel_i = Matrix((data_t *) arr_recurrent_kernel_i, NUM_UNITS, NUM_UNITS);
    Matrix recurrent_kernel_f = Matrix((data_t *) arr_recurrent_kernel_f, NUM_UNITS, NUM_UNITS);
    Matrix recurrent_kernel_c = Matrix((data_t *) arr_recurrent_kernel_c, NUM_UNITS, NUM_UNITS); 
    Matrix recurrent_kernel_o = Matrix((data_t *) arr_recurrent_kernel_o, NUM_UNITS, NUM_UNITS);

    data_t arr_bias_i[NUM_UNITS] = LSTM_BIAS_I;
    data_t arr_bias_f[NUM_UNITS] = LSTM_BIAS_F;
    data_t arr_bias_c[NUM_UNITS] = LSTM_BIAS_C;
    data_t arr_bias_o[NUM_UNITS] = LSTM_BIAS_O;
    Matrix bias_i = Matrix((data_t *)arr_bias_i, 1, NUM_UNITS);
    Matrix bias_f = Matrix((data_t *)arr_bias_f, 1, NUM_UNITS);
    Matrix bias_c = Matrix((data_t *)arr_bias_c, 1, NUM_UNITS);
    Matrix bias_o = Matrix((data_t *)arr_bias_o, 1, NUM_UNITS);
    LSTM_weight w_lstm = LSTM_weight( 
        &kernel_i, &kernel_f, &kernel_c, &kernel_o,
        &recurrent_kernel_i, &recurrent_kernel_f, &recurrent_kernel_c, &recurrent_kernel_o,
        &bias_i, &bias_f, &bias_c, &bias_o
    );

    // dense层参数初始化
    data_t arr_dense_kernel[NUM_UNITS][NUM_UNITS] = W_DENSE_KERNEL;
    Matrix dense_kernel = Matrix((data_t *)arr_dense_kernel, NUM_UNITS, NUM_UNITS);
    data_t arr_dense_bias[NUM_UNITS] = W_DENSE_BIAS;
    Matrix dense_bias = Matrix((data_t *)arr_dense_bias, 1, NUM_UNITS);

    // softmax层参数初始化
    data_t arr_softmax_kernel[NUM_UNITS][10] = W_SOFTMAX_KERNEL;
    Matrix softmax_kernel = Matrix((data_t *)arr_softmax_kernel, NUM_UNITS, 10);
    data_t arr_softmax_bias[10] = W_SOFTMAX_BIAS;
    Matrix softmax_bias = Matrix((data_t *)arr_softmax_bias, 1, 10); 
};