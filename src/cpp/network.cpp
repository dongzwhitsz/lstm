#include "../include/network.h"

Network::Network()
{  
}

// #define VAR  (softmax.output)
//     for(int i = 0; i<VAR.n_row; ++i)
//     {
//         for(int j = 0; j < VAR.n_col; ++j)
//         {
//             cout << VAR.get_value(i, j) << " " ;
//         }
//         cout << endl;
//     }
//     cout << "n_row: " << VAR.n_row << " n_col: " << VAR.n_col<< endl;
//     cout << endl;

void Network::forward()
{
    this->lstm.forward(&input);  // input: TIME_STEPS*TIME_STEPS 输出 1 x 256    
    this->dense.forward(&this->lstm.h); // dense NUM_UNITS*NUM_UNITS 输出 1 x 256    
    this->softmax.forward(&this->dense.output); // 输出 1 * 10    
    this->softmax.output.softmax();
}