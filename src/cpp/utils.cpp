#include "../include/utils.h"
#include <cmath>

data_t Vector::dotproduct(Vector &A, Vector &B)
{
    assert( A.size == B.size );
    data_t result = 0;
    for( int i = 0; i < A.size; i++)
    {
        result += A.pdata[i] * B.pdata[i];
    }
    return result;
}

void Vector::softmax()
{
    data_t sum = 0;
    for(int i = 0; i < this->size; ++i)
    {
        data_t _v = this->get_value(i);
        _v = exp(_v);
        sum += _v;
        this->set_value(i, _v);
    }
    for(int i = 0; i < this->size; ++i)
    {
        this->set_value(i, this->get_value(i) / sum);
    }
}


/**************************   Matrix  ************************/
// 获取矩阵中坐标点的值
data_t Matrix::get_value(int row, int col)
{
    return *(this->pdata + this->n_col * row + col);
}

void Matrix::set_value(int row, int col, data_t value)
{
    *(this->pdata + row * this->n_col + col) = value;
}

void Matrix::merge(Matrix *m, data_t *p)
{
    
}

void Matrix::hard_sigmoid()
{
    // keras 使用hard sigmoid 方式 
    for( int i = 0; i < this->n_row; ++i)
    {
        for( int j = 0; j < this-> n_col; ++j)
        {
            data_t _v = this->get_value(i, j);
            _v = ((_v + 1)/2 < 1 )?(_v + 1) / 2: 1;
            _v = (_v > 0)? _v: 0; 
            this->set_value(i, j, _v);
        }
    }
}
void Matrix::tanh()
{
    for( int i = 0; i < this->n_row; ++i)
    {
        for( int j = 0; j < this-> n_col; ++j)
        {
            data_t _v = this->get_value(i, j);
            _v = (exp(_v) - exp(-_v)) / ((exp(_v) + exp(-_v)));
            this->set_value(i, j, _v);
        }
    }
}

void Matrix::relu()
{
    for( int i = 0; i < this->n_row; ++i)
    {
        for( int j = 0; j < this-> n_col; ++j)
        {
            data_t _v = this->get_value(i, j);
            _v = _v > 0 ? _v: 0;
            this->set_value(i, j, _v);
        }
    }
}

void Matrix::softmax()
{
    assert(this->n_row == 1);
    data_t sum = 0;
    for(int i = 0; i < this->n_col; ++i)
    {
        data_t _v = this->get_value(0, i);
        _v = exp(_v);
        sum += _v;
        this->set_value(0, i, _v);
    }
    for(int i = 0; i < this->n_col; ++i)
    {
        data_t _v = this->get_value(0, i);
        _v = _v / sum;
        this->set_value(0, i, _v);
    }
}

bool Matrix::matmul(Matrix *m, Matrix *out)
{
    if ((this->n_col != m->n_row) || (this->n_row != out->n_row) || (m->n_col != out->n_col))
    {
        return false;
    }
    for( int r = 0; r < out->n_row; ++r )
    {
        for( int c = 0; c < out->n_col; ++c )
        {
            data_t _v = 0;
            for(int i = 0; i < this->n_col; ++i)
            {
                _v += this->get_value(r, i) * m->get_value(i, c);
            }
            out->set_value(r, c, _v);
        }
    }
    return true;
}

bool Matrix::matadd(Matrix *m)
{
    if((this->n_col != m->n_col) || (this->n_row != m->n_row))
    {
        return false;
    }
    for( int i = 0; i < this->n_row; ++i)
    {
        for( int j = 0; j < this->n_col; ++ j)
        {
            data_t _v = this->get_value(i, j) + m->get_value(i, j);
            this->set_value(i, j, _v);
        }
    }
    return true;
}

bool Matrix::dotmul(Matrix *m)
{
    if((this->n_col != m->n_col) || (this->n_row != m->n_row))
    {
        return false;
    }else
    {
        for(int i = 0; i < this->n_row; ++i)
        {
            for(int j = 0; j < this->n_col; ++j)
            {
                data_t a = this->get_value(i, j);
                data_t b = m->get_value(i, j);
                this->set_value(i, j, a * b);
            }
        }
    }
}

Matrix Matrix::matcopy(data_t *pdata)
{
    Matrix m(pdata, this->n_row, this->n_col);
    for(int i = 0; i < m.n_row; i++)
    {
        for(int j = 0; j< m.n_col; j++)
        {
            m.set_value(i, j, this->get_value(i, j));
        }
    }
    return m;
}

Matrix Matrix::matcopy(Matrix *m)
{
    assert( this->n_col == m->n_col );
    assert( this->n_row == m->n_row );
    for(int i = 0; i < this->n_row; ++i)
    {
        for(int j = 0; j < this->n_col; ++j)
        {
            m->set_value(i, j, this->get_value(i, j));
        }
    }
}

void Matrix::xw_plus_b(Matrix *m, Matrix *out, Matrix *v)
{
    // assert(out->n_row == v->n_col);
    this->matmul(m, out);
    for(int i = 0; i < out->n_row; ++i)
    {
        for(int j = 0; j < out->n_col; ++j)
        {
            data_t _v = out->get_value(i, j);
            _v += v->get_value(1, i);
            out->set_value(i, j, _v);
        }
    }
}


int main01()
{
    //
}