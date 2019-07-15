
#include "../weight/dense_bias.h"
#include "../weight/dense_kernel.h"

#include "../weight/lstm_kernel_o.h"
#include "../weight/lstm_kernel_c.h"
#include "../weight/lstm_kernel_f.h"
#include "../weight/lstm_kernel_i.h"
#include "../weight/lstm_bias_c.h"
#include "../weight/lstm_bias_f.h"
#include "../weight/lstm_bias_i.h"
#include "../weight/lstm_bias_o.h"
#include "../weight/lstm_recurrent_kernel_c.h"
#include "../weight/lstm_recurrent_kernel_f.h"
#include "../weight/lstm_recurrent_kernel_i.h"
#include "../weight/lstm_recurrent_kernel_o.h"
#include "../weight/softmax_bias.h"
#include "../weight/softmax_kernel.h"

//#include "./img/0_5.h"
//#include "./img/1_0.h"
//#include "./img/2_4.h"
//#include "./img/0_5.h"
//#include "./img/1_0.h"
//#include "./img/3_1.h"
//#include "./img/4_9.h"
//#include "./img/5_2.h"
//#include "./img/6_1.h"
//#include "./img/7_3.h"
//#include "./img/8_1.h"
//#include "./img/9_4.h"


typedef float data_t;


#define IMG_S  28
#define NUM_UNITS  128
#define DENSE_UNITS 128

// 鏆村姏鏋氫妇缃戠粶鍚勫鍙傛暟
/************* LSTM params ****************/
data_t c[NUM_UNITS] = {0};
data_t h[NUM_UNITS] = {0};
data_t lstm_bias_c[NUM_UNITS] = LSTM_BIAS_C;
data_t lstm_bias_f[NUM_UNITS] = LSTM_BIAS_F;
data_t lstm_bias_i[NUM_UNITS] = LSTM_BIAS_I;
data_t lstm_bias_o[NUM_UNITS] = LSTM_BIAS_O;

data_t lstm_kernel_c[IMG_S][NUM_UNITS] = W_LSTM_KERNEL_C;
data_t lstm_kernel_f[IMG_S][NUM_UNITS] = W_LSTM_KERNEL_F;
data_t lstm_kernel_i[IMG_S][NUM_UNITS] = W_LSTM_KERNEL_I;
data_t lstm_kernel_o[IMG_S][NUM_UNITS] = W_LSTM_KERNEL_O;

data_t lstm_recurrent_kernel_c[NUM_UNITS][NUM_UNITS] = W_LSTM_RECURRENT_KERNEL_C;
data_t lstm_recurrent_kernel_f[NUM_UNITS][NUM_UNITS] = W_LSTM_RECURRENT_KERNEL_F;
data_t lstm_recurrent_kernel_i[NUM_UNITS][NUM_UNITS] = W_LSTM_RECURRENT_KERNEL_I;
data_t lstm_recurrent_kernel_o[NUM_UNITS][NUM_UNITS] = W_LSTM_RECURRENT_KERNEL_O;

/************* dense params ****************/
data_t dense_bias[DENSE_UNITS] = W_DENSE_BIAS;
data_t dense_kernel[NUM_UNITS][DENSE_UNITS] = W_DENSE_KERNEL;
// data_t dense_input[DENSE_UNITS] = {0};
data_t dense_output[DENSE_UNITS] = {0};

/************* softmax params ****************/
data_t softmax_bias[10] = W_SOFTMAX_BIAS;
data_t softmax_kernel[NUM_UNITS][10] = W_SOFTMAX_KERNEL;
// data_t softmax_input[10] = {0};
data_t softmax_output[10] = {0};


/******************** function prototype declare *********************************** */
void img_preprocess(data_t img[IMG_S][IMG_S]);
void lstm_forward(data_t img[IMG_S][IMG_S]);
void lstm_forward_once(data_t img_line[IMG_S]);
void dense_forward(data_t h[NUM_UNITS]);
void softmax_forward(data_t dense_output[DENSE_UNITS]);


/*********************** function realizaton ************************************* */
void top(data_t img[IMG_S][IMG_S], data_t output[10])
{
    // 棣栧厛棰勫鐞嗗浘锟�?
   img_preprocess(img);
   lstm_forward(img);       
   dense_forward(h);
   softmax_forward(dense_output);
   // output the result of the network;
   for(int i = 0; i < 10; ++i)
   {
       output[i] = softmax_output[i];
   }
}

//
//int main()
//{
//    // test bench for the top function in g++;
//    // data_t arr[28][28] = IMG_0_5;
//    // data_t arr[28][28] = IMG_1_0;
//    // data_t arr[28][28] = IMG_2_4;
//    // data_t arr[28][28] = IMG_3_1;
//    // data_t arr[28][28] = IMG_4_9;
//    data_t arr[28][28] = IMG_5_2;
//    // data_t arr[28][28] = IMG_6_1;
//    // data_t arr[28][28] = IMG_7_3;
//    // data_t arr[28][28] = IMG_8_1;
//    // data_t arr[28][28] = IMG_9_4;
//    data_t output[10] = {0};
//    top(arr, output);
//
//    for(int i = 0; i < 10; i ++)
//    {
//        cout <<  i << ": " << output[i] << endl;
//    }
//    cout << endl;
//    return 0;
//}

void img_preprocess(data_t img[IMG_S][IMG_S])
{
    for(int i = 0; i < IMG_S; i++)
    {
        for(int j = 0; j < IMG_S; j++)
        {
            img[i][j] = img[i][j] / 128.0 - 1;
        }
    }
}


void lstm_forward(data_t img[IMG_S][IMG_S])
{
    for(int i  = 0; i < NUM_UNITS; ++i)
    {
        c[i] = 0;
        h[i] = 0;
    }
    for(int i = 0; i < IMG_S; ++i)
    {
        lstm_forward_once(img[i]);
    }
}


void lstm_forward_once(data_t img_line[IMG_S])
{
    /************** forget gate *************** */
    data_t arr1[NUM_UNITS] = {0};
    // img_line 锟�? 鏉冮噸浣滅敤
    for(int i = 0; i < NUM_UNITS; ++i)
    {
        for(int j = 0; j < IMG_S; ++j)
        {
             arr1[i] += img_line[j] * lstm_kernel_f[j][i];
        }
    }
    data_t arr2[NUM_UNITS] = {0};
    // h matmul with weights
    for(int i = 0; i < NUM_UNITS; ++i)
    {
        for(int j = 0; j < NUM_UNITS; ++j)
        {
            arr2[i] += h[j] * lstm_recurrent_kernel_f[j][i];
        }
    }
    data_t arr3[NUM_UNITS] = {0};
    // store the hard sigmoid result in the arr3
    for(int i = 0; i < NUM_UNITS; ++i)
    {
        arr3[i] = arr1[i] + arr2[i]+ lstm_bias_f[i];
        // arr3 add the bias
        arr3[i] = 0.2 * arr3[i] + 0.5;
        if (arr3[i] >2.5)
            arr3[i] = 1;
        else if (arr3[i] < -2.5)
            arr3[i] = 0;
    }

    /************** input gate *************** */
    data_t arr4[NUM_UNITS] = {0};
    // img_line 锟�? 鏉冮噸浣滅敤
    for(int i = 0; i < NUM_UNITS; ++i)
    {
        for(int j = 0; j < IMG_S; ++j)
        {
             arr4[i] += img_line[j] * lstm_kernel_i[j][i];
        }
    }
    data_t arr5[NUM_UNITS] = {0};
    // h matmul with weights
    for(int i = 0; i < NUM_UNITS; ++i)
    {
        for(int j = 0; j < NUM_UNITS; ++j)
        {
            arr5[i] += h[j] * lstm_recurrent_kernel_i[j][i];
        }
    }
    data_t arr6[NUM_UNITS] = {0};
    // store the hard sigmoid result in the arr6
    for(int i = 0; i < NUM_UNITS; ++i)
    {
        // add the bias to the arr6
        arr6[i] = arr4[i] + arr5[i] + lstm_bias_i[i];
        arr6[i] = 0.2 * arr6[i] + 0.5;
        if (arr6[i] > 2.5)
            arr6[i] = 1;
        else if (arr6[i] < -2.5)
            arr6[i] = 0;
    }

 /************** candidate gate *************** */
    data_t arr7[NUM_UNITS] = {0};
    // img_line 锟�? 鏉冮噸浣滅敤
    for(int i = 0; i < NUM_UNITS; ++i)
    {
        for(int j = 0; j < IMG_S; ++j)
        {
             arr7[i] += img_line[j] * lstm_kernel_c[j][i];
        }
    }
    data_t arr8[NUM_UNITS] = {0};
    // h matmul with weights
    for(int i = 0; i < NUM_UNITS; ++i)
    {
        for(int j = 0; j < NUM_UNITS; ++j)
        {
            arr8[i] += h[j] * lstm_recurrent_kernel_c[j][i];
        }
    }
    data_t arr9[NUM_UNITS] = {0};
    // store the hard sigmoid result in the arr9
    for(int i = 0; i < NUM_UNITS; ++i)
    {
        // add the bias to the arr9
        arr9[i] = arr7[i] + arr8[i] + lstm_bias_c[i];
        // arr9[i] = tanh(arr9[i]);
        arr9[i] = 0.8 * arr9[i];
        if (arr9[i] > 1)
            arr9[i] = 1;
        else if (arr9[i] < -1)
            arr9[i] = -1;
    }

    /************** output gate *************** */
    data_t arr10[NUM_UNITS] = {0};
    // img_line 锟�? 鏉冮噸浣滅敤
    for(int i = 0; i < NUM_UNITS; ++i)
    {
        for(int j = 0; j < IMG_S; ++j)
        {
             arr10[i] += img_line[j] * lstm_kernel_o[j][i];
        }
    }
    data_t arr11[NUM_UNITS] = {0};
    // h matmul with weights
    for(int i = 0; i < NUM_UNITS; ++i)
    {
        for(int j = 0; j < NUM_UNITS; ++j)
        {
            arr11[i] += h[j] * lstm_recurrent_kernel_o[j][i];
        }
    }
    data_t arr12[NUM_UNITS] = {0};
    // store the hard sigmoid result in the arr12
    for(int i = 0; i < NUM_UNITS; ++i)
    {
        arr12[i] = arr10[i] + arr11[i] + lstm_bias_o[i];
        // add the bias to the arr12
        arr12[i] = arr12[i] * 0.2 + 0.5;
        if ( arr12[i] > 2.5)
        {
            arr12[i] = 1;
        }else if( arr12[i] < -2.5)
        {
            arr12[i] = 0;
        }
    }

    // get the c and h
    for(int i = 0; i < NUM_UNITS; ++i)
    {
        c[i] = c[i] * arr3[i];
    }

    for(int i = 0; i < NUM_UNITS; ++i)
    {
        c[i] = c[i] + arr6[i] * arr9[i];
    }

    data_t arr13[NUM_UNITS] = {0};
    for(int i = 0; i < NUM_UNITS; ++i)
    {
        // arr13[i] = tanh(c[i]);
        arr13[i] = c[i] * 0.8;
        if(arr13[i] > 1)
        {
            arr13[i] = 1;
        }else if(arr13[i] < -1)
        {
            arr13[i] = -1;
        }
    }

    for(int i = 0; i < NUM_UNITS; ++i)
    {
        h[i] = arr13[i] * arr12[i];
    }
}


void dense_forward(data_t h[NUM_UNITS])
{
    for(int i = 0; i < DENSE_UNITS; ++i)
    {
        dense_output[i] = 0;
        for(int j = 0; j < NUM_UNITS; ++j)
            dense_output[i] += dense_kernel[j][i] * h[j];
    }
    // add bias
    for(int i = 0; i < DENSE_UNITS; ++i)
    {
        dense_output[i] += dense_bias[i];
        // relu
        if(dense_output[i] < 0)
        {
            dense_output[i] = 0;
        }
    }
}

void softmax_forward(data_t dense_output[DENSE_UNITS])
{
    for(int i = 0; i < 10; ++i)
    {
        softmax_output[i] = 0;
        for(int j = 0; j < DENSE_UNITS; ++j)
            softmax_output[i] += softmax_kernel[j][i] * dense_output[j];
    }
    // add bias
    for(int i = 0; i < 10; ++i)
    {
        softmax_output[i] += softmax_bias[i];
    }
//    // don't need to do the exp in forward step;
//    data_t s = 0;
//    for(int i = 0 ; i < 10; i ++)
//    {
//        softmax_output[i] = exp(softmax_output[i]);
//        s += softmax_output[i];
//    }
//    for(int i = 0; i < 10; i ++)
//    {
//        softmax_output[i] = softmax_output[i] / s;
//    }
}
