#pragma once
#include <iostream> 
#include <assert.h>
using namespace std;

typedef float data_t;

class Data
{
public:
};

// 向量可以使用二维矩阵代替
class Vector: public Data
{
public:
    Vector(data_t *pdata, int size)
    {
        this->pdata = pdata;
        this->size = size;
    }
    static data_t dotproduct(Vector &A, Vector &B);
    data_t get_value(int pos)
    {
        return this->pdata[pos];
    }
    void set_value(int pos, data_t value)
    {
        this->pdata[pos] = value;
    }
    void softmax();
public:
    // 矩阵的数据统一表示成一级指针形式。
    data_t *pdata;
    int size;
};

class Matrix: public Data
{
    // 将向量表示为 (1, size)的矩阵
public:
    Matrix(data_t *pdata, int n_row, int n_col)
    {
        this->pdata = pdata;
        this->n_row = n_row;
        this->n_col = n_col;
    }
    // 重写Matrix的拷贝构造函数
    // Matrix(const Matrix&m);没有 new是真不好做
    Matrix matcopy(data_t *pdata);
    Matrix matcopy(Matrix *m);

    bool matadd(Matrix *m);
    bool dotmul(Matrix *m); // 与矩阵m按元素做乘法
    bool matmul(Matrix *m, Matrix *out); // 矩阵乘法
    void xw_plus_b(Matrix *m, Matrix *out, Matrix *v);
    void merge(Matrix *m, data_t *p);

    data_t get_value(int row, int col);
    void set_value(int row, int col, data_t value);

    void hard_sigmoid();
    void relu();
    void tanh();
    void softmax();
public:
    int n_row;
    int n_col;    
private:
    data_t *pdata;
};

