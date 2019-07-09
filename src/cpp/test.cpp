#include "../include/network.h"
#include "../../input/mnist/train/0_5.h"
#include "../../input/mnist/train/1_0.h"
#include "../../input/mnist/train/2_4.h"
#include "../../input/mnist/train/3_1.h"
#include "../../input/mnist/train/4_9.h"
#include "../../input/mnist/train/5_2.h"
#include "../include/utils.h"
#include "../include/dense.h"
#include "../include/lstm.h"
#include<iostream>
using namespace std;

int main()
{
    // data_t arr[28][28] = IMG_0_5;
    // data_t arr[28][28] = IMG_1_0;
    // data_t arr[28][28] = IMG_2_4;
    // data_t arr[28][28] = IMG_3_1;
    data_t arr[28][28] = IMG_4_9;
    for(int i = 0; i < 28; ++i)
    {
        for(int j = 0; j < 28; ++j)
        {
            arr[i][j] = arr[i][j] /128 - 1;
        }
    }

    Matrix input = Matrix((data_t *)arr, 28, 28);
    Network network;
    network.input = input;
    network.forward();
    for(int i = 0; i < network.output.n_row; ++i)
    {
        for(int j = 0; j < network.output.n_col; ++j)
        {
            cout << network.output.get_value(i, j) << " ";
        }
        cout << endl;
    }

}
