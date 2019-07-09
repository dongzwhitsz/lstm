#pragma once
#include "utils.h"
class Softmax
{
public:
    Matrix *weight;
    Matrix *bias;
    // 为dense层分配内存
    data_t arr_output[1][10];
    Matrix output = Matrix((data_t *)arr_output, 1, 10);

public:
    Softmax(Matrix *weight, Matrix *bias)
    {
        this->weight = weight;
        this->bias = bias;
    }
    void forward(Matrix *input)
    {
        input->xw_plus_b(this->weight, &output, bias);
        output.softmax();
    }
};