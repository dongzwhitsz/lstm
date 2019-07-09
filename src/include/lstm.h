#pragma once
#include "../include/utils.h"
#include <assert.h>

#define TIME_STEPS 28
#define INPUT_DIM 28
#define NUM_UNITS 256

class  LSTM_weight
{
public:
    // 与x相乘
    Matrix *kernel_i;
    Matrix *kernel_f;
    Matrix *kernel_c;
    Matrix *kernel_o;
    // 与h相乘
    Matrix *recurrent_kernel_i;
    Matrix *recurrent_kernel_f;
    Matrix *recurrent_kernel_c; 
    Matrix *recurrent_kernel_o;
    Matrix *bias_i;
    Matrix *bias_f;
    Matrix *bias_c;
    Matrix *bias_o;
public:
LSTM_weight(
    Matrix *kernel_i,
    Matrix *kernel_f,
    Matrix *kernel_c,
    Matrix *kernel_o,
    // 与h相乘
    Matrix *recurrent_kernel_i,
    Matrix *recurrent_kernel_f,
    Matrix *recurrent_kernel_c, 
    Matrix *recurrent_kernel_o,
    Matrix *bias_i,
    Matrix *bias_f,
    Matrix *bias_c,
    Matrix *bias_o)
    {
        this->kernel_i = kernel_i;
        this->kernel_f = kernel_f;
        this->kernel_c = kernel_c;
        this->kernel_o = kernel_o;
        // 与h相乘
        this->recurrent_kernel_i = recurrent_kernel_i;
        this->recurrent_kernel_f = recurrent_kernel_f;
        this->recurrent_kernel_c = recurrent_kernel_c;   
        this->recurrent_kernel_o = recurrent_kernel_o;
        this->bias_i = bias_i;
        this->bias_f = bias_f;
        this->bias_c = bias_c;
        this->bias_o = bias_o;
    }
};

class LSTM
{
public:
    // 用权重初始化一个LSTM
    LSTM(LSTM_weight *weights)
    {
        this->weights = weights;
        // this->h = h;
        // this->c = c;
        
    }
    Matrix forward(Matrix *x_in);
    // Matrix forward(Matrix *x_in, Matrix *h, Matrix *c);
    void forward_once(Matrix *x_in);
    void set_zeros(); // 将当前 h 和 c 的状态设置为全0

public:    
// lstm的状态初始化
    data_t arr_h[1][NUM_UNITS] = {0};
    data_t arr_c[1][NUM_UNITS] = {0};
    Matrix h = Matrix((data_t *)arr_h, 1, NUM_UNITS);
    Matrix c = Matrix((data_t *)arr_c, 1, NUM_UNITS);
    // Matrix *h; // h即是输出
    // Matrix *c;
 

private:    
    LSTM_weight *weights;// keras的训练权重数组

    data_t v_forget_1[1][NUM_UNITS]; //存放与x做乘法的结果 //  x[1][INPUT_DIM]  * [INPUT_DIM][NUM_UNITS]  = [1][NUM_UNITS]
    data_t v_forget_2[1][NUM_UNITS]; //存放与h做乘法的结果  h[1][NUM_UNITS] * [NUM_UNITS][NUM_UNITS] = [1][NUM_UNITS]
    Matrix forget_1 = Matrix((data_t *)v_forget_1, 1, NUM_UNITS);
    Matrix forget_2 = Matrix((data_t *)v_forget_2, 1, NUM_UNITS);

    data_t v_input_1[1][NUM_UNITS]; //存放与x做乘法的结果
    data_t v_input_2[1][NUM_UNITS]; //存放与h做乘法的结果
    Matrix input_1 = Matrix((data_t *)v_input_1, 1, NUM_UNITS);
    Matrix input_2 = Matrix((data_t *)v_input_2, 1, NUM_UNITS);

    data_t v_carry_1[1][NUM_UNITS]; //存放与x做乘法的结果
    data_t v_carry_2[1][NUM_UNITS]; //存放与h做乘法的结果
    Matrix carry_1 = Matrix((data_t *)v_carry_1, 1, NUM_UNITS);
    Matrix carry_2 = Matrix((data_t *)v_carry_2, 1, NUM_UNITS);

    data_t v_output_1[1][NUM_UNITS]; //存放与x做乘法的结果
    data_t v_output_2[1][NUM_UNITS]; //存放与h做乘法的结果
    Matrix output_1 = Matrix((data_t *)v_output_1, 1, NUM_UNITS);
    Matrix output_2 = Matrix((data_t *)v_output_2, 1, NUM_UNITS);

    data_t v_cell_1[1][NUM_UNITS];
    Matrix cell_1 = Matrix((data_t *) v_cell_1, 1, NUM_UNITS);

    data_t v_cell_2[1][NUM_UNITS];
    Matrix cell_2 = Matrix((data_t *) v_cell_2, 1, NUM_UNITS);

    // 当前 timestep的输入暂存缓冲矩阵
    data_t arr_p[1][INPUT_DIM];
    Matrix p_in = Matrix((data_t *)arr_p, 1, INPUT_DIM);
        
private:
    void copy_matrix_line(Matrix *src, Matrix *dst, int pos);

};