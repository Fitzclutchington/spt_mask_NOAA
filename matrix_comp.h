#ifndef MATRIX_COMP_H_
#define MATRIX_COMP_H_

namespace acspo {

template <typename T, typename S>
matrix<bool> operator==(const matrix<T> &mat1, const matrix<S> &mat2)
{
    if (mat1.size() != mat2.size()) {
        throw std::runtime_error("dimension mismatch");
    }
    unsigned int rows = mat1.rows(), cols = mat1.cols(), elem = mat1.elem();
    matrix<bool> ret(rows, cols);
    bool *rptr = ret.ptr();
    const T *m1ptr = mat1.ptr();
    const T *m2ptr = mat2.ptr();
    #pragma omp parallel for
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = m1ptr[i] == m2ptr[i];
    }
    return ret;
}

template <typename T, typename S>
matrix<bool> operator!=(const matrix<T> &mat1, const matrix<S> &mat2)
{
    if (mat1.size() != mat2.size()) {
        throw std::runtime_error("dimension mismatch");
    }
    unsigned int rows = mat1.rows(), cols = mat1.cols(), elem = mat1.elem();
    matrix<bool> ret(rows, cols);
    bool *rptr = ret.ptr();
    const T *m1ptr = mat1.ptr();
    const T *m2ptr = mat2.ptr();
    #pragma omp parallel for
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = m1ptr[i] != m2ptr[i];
    }
    return ret;
}

template <typename T, typename S>
matrix<bool> operator<(const matrix<T> &mat1, const matrix<S> &mat2)
{
    if (mat1.size() != mat2.size()) {
        throw std::runtime_error("dimension mismatch");
    }
    unsigned int rows = mat1.rows(), cols = mat1.cols(), elem = mat1.elem();
    matrix<bool> ret(rows, cols);
    bool *rptr = ret.ptr();
    const T *m1ptr = mat1.ptr();
    const T *m2ptr = mat2.ptr();
    #pragma omp parallel for
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = m1ptr[i] < m2ptr[i];
    }
    return ret;
}

template <typename T, typename S>
matrix<bool> operator<=(const matrix<T> &mat1, const matrix<S> &mat2)
{
    if (mat1.size() != mat2.size()) {
        throw std::runtime_error("dimension mismatch");
    }
    unsigned int rows = mat1.rows(), cols = mat1.cols(), elem = mat1.elem();
    matrix<bool> ret(rows, cols);
    bool *rptr = ret.ptr();
    const T *m1ptr = mat1.ptr();
    const T *m2ptr = mat2.ptr();
    #pragma omp parallel for
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = m1ptr[i] <= m2ptr[i];
    }
    return ret;
}

template <typename T, typename S>
matrix<bool> operator>(const matrix<T> &mat1, const matrix<S> &mat2)
{
    if (mat1.size() != mat2.size()) {
        throw std::runtime_error("dimension mismatch");
    }
    unsigned int rows = mat1.rows(), cols = mat1.cols(), elem = mat1.elem();
    matrix<bool> ret(rows, cols);
    bool *rptr = ret.ptr();
    const T *m1ptr = mat1.ptr();
    const T *m2ptr = mat2.ptr();
    #pragma omp parallel for
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = m1ptr[i] > m2ptr[i];
    }
    return ret;
}

template <typename T, typename S>
matrix<bool> operator>=(const matrix<T> &mat1, const matrix<S> &mat2)
{
    if (mat1.size() != mat2.size()) {
        throw std::runtime_error("dimension mismatch");
    }
    unsigned int rows = mat1.rows(), cols = mat1.cols(), elem = mat1.elem();
    matrix<bool> ret(rows, cols);
    bool *rptr = ret.ptr();
    const T *m1ptr = mat1.ptr();
    const T *m2ptr = mat2.ptr();
    #pragma omp parallel for
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = m1ptr[i] >= m2ptr[i];
    }
    return ret;
}

template <typename T, typename S>
matrix<bool> operator==(const matrix<T> &mat, const S &val)
{
    unsigned int rows = mat.rows(), cols = mat.cols(), elem = mat.elem();
    matrix<bool> ret(rows, cols);
    bool *rptr = ret.ptr();
    const T *mptr = mat.ptr();
    #pragma omp parallel for
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = mptr[i] == val;
    }
    return ret;
}

template <typename T, typename S>
matrix<bool> operator!=(const matrix<T> &mat, const S &val)
{
    unsigned int rows = mat.rows(), cols = mat.cols(), elem = mat.elem();
    matrix<bool> ret(rows, cols);
    bool *rptr = ret.ptr();
    const T *mptr = mat.ptr();
    #pragma omp parallel for
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = mptr[i] != val;
    }
    return ret;
}

template <typename T, typename S>
matrix<bool> operator<(const matrix<T> &mat, const S &val)
{
    unsigned int rows = mat.rows(), cols = mat.cols(), elem = mat.elem();
    matrix<bool> ret(rows, cols);
    bool *rptr = ret.ptr();
    const T *mptr = mat.ptr();
    #pragma omp parallel for
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = mptr[i] < val;
    }
    return ret;
}

template <typename T, typename S>
matrix<bool> operator<=(const matrix<T> &mat, const S &val)
{
    unsigned int rows = mat.rows(), cols = mat.cols(), elem = mat.elem();
    matrix<bool> ret(rows, cols);
    bool *rptr = ret.ptr();
    const T *mptr = mat.ptr();
    #pragma omp parallel for
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = mptr[i] <= val;
    }
    return ret;
}

template <typename T, typename S>
matrix<bool> operator>(const matrix<T> &mat, const S &val)
{
    unsigned int rows = mat.rows(), cols = mat.cols(), elem = mat.elem();
    matrix<bool> ret(rows, cols);
    bool *rptr = ret.ptr();
    const T *mptr = mat.ptr();
    #pragma omp parallel for
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = mptr[i] > val;
    }
    return ret;
}

template <typename T, typename S>
matrix<bool> operator>=(const matrix<T> &mat, const S &val)
{
    unsigned int rows = mat.rows(), cols = mat.cols(), elem = mat.elem();
    matrix<bool> ret(rows, cols);
    bool *rptr = ret.ptr();
    const T *mptr = mat.ptr();
    #pragma omp parallel for
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = mptr[i] >= val;
    }
    return ret;
}

template <typename T, typename S>
matrix<bool> operator==(const S &val, const matrix<T> &mat)
{
    return mat == val;
}

template <typename T, typename S>
matrix<bool> operator!=(const S &val, const matrix<T> &mat)
{
    return mat != val;
}

template <typename T, typename S>
matrix<bool> operator<(const S &val, const matrix<T> &mat)
{
    return mat > val;
}

template <typename T, typename S>
matrix<bool> operator<=(const S &val, const matrix<T> &mat)
{
    return mat >= val;
}

template <typename T, typename S>
matrix<bool> operator>(const S &val, const matrix<T> &mat)
{
    return mat < val;
}

template <typename T, typename S>
matrix<bool> operator>=(const S &val, const matrix<T> &mat)
{
    return mat <= val;
}

}

#endif
