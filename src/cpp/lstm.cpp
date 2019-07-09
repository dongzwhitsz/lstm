#include "../include/lstm.h"
#include <cmath>


void LSTM::copy_matrix_line(Matrix *src, Matrix *dst, int pos)
{
    assert(dst->n_row == 1);
    for(int i = 0; i < INPUT_DIM; ++i)
    {
        dst->set_value(0, i, src->get_value(pos, i));
    }
}

void LSTM::set_zeros()
{
    for(int i = 0; i < h.n_row; ++i)
    {
        for(int j = 0; j < h.n_col; ++j)
        {
            h.set_value(i, j, 0);
        }
    }
    for(int i = 0; i < c.n_row; ++i)
    {
        for(int j = 0; j < c.n_col; ++j)
        {
            c.set_value(i, j, 0);
        }
    }
}

Matrix LSTM::forward(Matrix *x_in)
{
    this->set_zeros();
    for(int i = 0; i < TIME_STEPS; ++i)
    {
        // #define VAR  (this->h)
        // for(int i = 0; i<VAR.n_row; ++i)
        // {
        //     for(int j = 0; j < VAR.n_col; ++j)
        //     {
        //         cout << VAR.get_value(i, j) << " " ;
        //     }
        //     cout << endl;
        // }
        // cout << "n_row: " << VAR.n_row << " n_col: " << VAR.n_col<< endl;
        // cout << "i: " << i << endl;

        this->copy_matrix_line(x_in, &p_in, i);
        forward_once(&p_in);
    }
}

void LSTM::forward_once(Matrix *x_in)
{
    assert( x_in->n_row == 1);
    // 遗忘门  此时输入应当为 (1, 28) h_state应该为(1, 256);
    x_in->matmul(weights->kernel_f, &forget_1);
    h.matmul(weights->recurrent_kernel_f, &forget_2);
    forget_1.matadd(&forget_2);
    forget_1.matadd(weights->bias_f);
    forget_1.hard_sigmoid(); // forget_1 就是遗忘门的输出；

    // 输入门
    x_in->matmul(weights->kernel_i, &input_1);
    h.matmul(weights->recurrent_kernel_i, &input_2);
    input_1.matadd(&input_2);
    input_1.matadd(weights->bias_i);
    input_1.hard_sigmoid(); // input_1 就是输入门的输出

    //carry 门
    x_in->matmul(weights->kernel_c, &carry_1);
    h.matmul(weights->recurrent_kernel_c, &carry_2);
    carry_1.matadd(&carry_2);
    carry_1.matadd(weights->bias_c);
    carry_1.tanh(); // carry_1 就是carry门的输出

    //输出门
    x_in->matmul(weights->kernel_o, &output_1);
    h.matmul(weights->recurrent_kernel_o, &output_2);
    output_1.matadd(&output_2);
    output_1.matadd(weights->bias_o);
    output_1.hard_sigmoid(); // output_1 就是输出门的输出

    // cell 状态的更新
    c.dotmul(&forget_1);
    input_1.dotmul(&carry_1);
    c.matadd(&input_1);  // 此时的c为更新状态之后的c状态；

    // lstm的forward 的运行结果
    c.matcopy(&h);
    h.tanh();
    h.dotmul(&output_1);
}


