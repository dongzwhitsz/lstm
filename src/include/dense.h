#pragma once
#include "utils.h"
class Dense
{
public:
    Matrix *weight;
    Matrix *bias;
    // 为dense层分配内存
    data_t arr_output[1][NUM_UNITS];
    Matrix output = Matrix((data_t *)arr_output, 1, NUM_UNITS);

public:
    Dense(Matrix *weight, Matrix *bias)
    {
        this->weight = weight;
        this->bias = bias;
    }
    void forward(Matrix *input)
    {
        input->xw_plus_b(this->weight, &output, bias);
        output.relu();
    }
};