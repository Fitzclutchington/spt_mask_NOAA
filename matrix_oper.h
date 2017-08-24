#ifndef MATRIX_OPER_H_
#define MATRIX_OPER_H_

namespace acspo {

template <typename T>
matrix<T> & operator+=(matrix<T> &mat1, const matrix<T> &mat2)
{
    if (mat1.size() != mat2.size()) {
        throw std::runtime_error("dimension mismatch");
    }
    unsigned int elem = mat1.elem();
    T *m1ptr = mat1.ptr();
    const T *m2ptr = mat2.ptr();
    #pragma omp parallel for simd
    for (unsigned int i = 0; i < elem; i++) {
        m1ptr[i] += m2ptr[i];
    }
    return mat1;
}

template <typename T>
matrix<T> & operator-=(matrix<T> &mat1, const matrix<T> &mat2)
{
    if (mat1.size() != mat2.size()) {
        throw std::runtime_error("dimension mismatch");
    }
    unsigned int elem = mat1.elem();
    T *m1ptr = mat1.ptr();
    const T *m2ptr = mat2.ptr();
    #pragma omp parallel for simd
    for (unsigned int i = 0; i < elem; i++) {
        m1ptr[i] -= m2ptr[i];
    }
    return mat1;
}

template <typename T>
matrix<T> & operator*=(matrix<T> &mat1, const matrix<T> &mat2)
{
    if (mat1.size() != mat2.size()) {
        throw std::runtime_error("dimension mismatch");
    }
    unsigned int elem = mat1.elem();
    T *m1ptr = mat1.ptr();
    const T *m2ptr = mat2.ptr();
    #pragma omp parallel for simd
    for (unsigned int i = 0; i < elem; i++) {
        m1ptr[i] *= m2ptr[i];
    }
    return mat1;
}

template <typename T>
matrix<T> & operator/=(matrix<T> &mat1, const matrix<T> &mat2)
{
    if (mat1.size() != mat2.size()) {
        throw std::runtime_error("dimension mismatch");
    }
    unsigned int elem = mat1.elem();
    T *m1ptr = mat1.ptr();
    const T *m2ptr = mat2.ptr();
    #pragma omp parallel for simd
    for (unsigned int i = 0; i < elem; i++) {
        m1ptr[i] /= m2ptr[i];
    }
    return mat1;
}

template <typename T, typename S>
matrix<T> & operator+=(matrix<T> &mat, const S &val)
{
    unsigned int elem = mat.elem();
    T *mptr = mat.ptr();
    #pragma omp parallel for simd
    for (unsigned int i = 0; i < elem; i++) {
        mptr[i] += val;
    }
    return mat;
}

template <typename T, typename S>
matrix<T> & operator-=(matrix<T> &mat, const S &val)
{
    unsigned int elem = mat.elem();
    T *mptr = mat.ptr();
    #pragma omp parallel for simd
    for (unsigned int i = 0; i < elem; i++) {
        mptr[i] -= val;
    }
    return mat;
}

template <typename T, typename S>
matrix<T> & operator*=(matrix<T> &mat, const S &val)
{
    unsigned int elem = mat.elem();
    T *mptr = mat.ptr();
    #pragma omp parallel for simd
    for (unsigned int i = 0; i < elem; i++) {
        mptr[i] *= val;
    }
    return mat;
}

template <typename T, typename S>
matrix<T> & operator/=(matrix<T> &mat, const S &val)
{
    unsigned int elem = mat.elem();
    T *mptr = mat.ptr();
    #pragma omp parallel for simd
    for (unsigned int i = 0; i < elem; i++) {
        mptr[i] /= val;
    }
    return mat;
}

template <typename T>
matrix<T> operator-(const matrix<T> &mat)
{
    unsigned int elem = mat.elem();
    matrix<T> ret(mat.size());
    T *rptr = ret.ptr();
    const T *mptr = mat.ptr();
    #pragma omp parallel for simd
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = -mptr[i];
    }
    return ret;
}

template <typename T>
matrix<T> operator+(const matrix<T> &mat1, const matrix<T> &mat2)
{
    if (mat1.size() != mat2.size()) {
        throw std::runtime_error("dimension mismatch");
    }
    unsigned int elem = mat1.elem();
    matrix<T> ret(mat1.size());
    T *rptr = ret.ptr();
    const T *m1ptr = mat1.ptr();
    const T *m2ptr = mat2.ptr();
    #pragma omp parallel for simd
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = m1ptr[i] + m2ptr[i];
    }
    return ret;
}

template <typename T>
matrix<T> operator-(const matrix<T> &mat1, const matrix<T> &mat2)
{
    if (mat1.size() != mat2.size()) {
        throw std::runtime_error("dimension mismatch");
    }
    unsigned int elem = mat1.elem();
    matrix<T> ret(mat1.size());
    T *rptr = ret.ptr();
    const T *m1ptr = mat1.ptr();
    const T *m2ptr = mat2.ptr();
    #pragma omp parallel for simd
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = m1ptr[i] - m2ptr[i];
    }
    return ret;
}

template <typename T>
matrix<T> operator*(const matrix<T> &mat1, const matrix<T> &mat2)
{
    if (mat1.size() != mat2.size()) {
        throw std::runtime_error("dimension mismatch");
    }
    unsigned int elem = mat1.elem();
    matrix<T> ret(mat1.size());
    T *rptr = ret.ptr();
    const T *m1ptr = mat1.ptr();
    const T *m2ptr = mat2.ptr();
    #pragma omp parallel for simd
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = m1ptr[i] * m2ptr[i];
    }
    return ret;
}

template <typename T>
matrix<T> operator/(const matrix<T> &mat1, const matrix<T> &mat2)
{
    if (mat1.size() != mat2.size()) {
        throw std::runtime_error("dimension mismatch");
    }
    unsigned int elem = mat1.elem();
    matrix<T> ret(mat1.size());
    T *rptr = ret.ptr();
    const T *m1ptr = mat1.ptr();
    const T *m2ptr = mat2.ptr();
    #pragma omp parallel for simd
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = m1ptr[i] / m2ptr[i];
    }
    return ret;
}

template <typename T, typename S>
matrix<T> operator+(const matrix<T> &mat, const S &val)
{
    unsigned int elem = mat.elem();
    matrix<T> ret(mat.size());
    T *rptr = ret.ptr();
    const T *mptr = mat.ptr();
    #pragma omp parallel for simd
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = mptr[i] + val;
    }
    return ret;
}

template <typename T, typename S>
matrix<T> operator-(const matrix<T> &mat, const S &val)
{
    unsigned int elem = mat.elem();
    matrix<T> ret(mat.size());
    T *rptr = ret.ptr();
    const T *mptr = mat.ptr();
    #pragma omp parallel for simd
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = mptr[i] - val;
    }
    return ret;
}

template <typename T, typename S>
matrix<T> operator*(const matrix<T> &mat, const S &val)
{
    unsigned int elem = mat.elem();
    matrix<T> ret(mat.size());
    T *rptr = ret.ptr();
    const T *mptr = mat.ptr();
    #pragma omp parallel for simd
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = mptr[i] * val;
    }
    return ret;
}

template <typename T, typename S>
matrix<T> operator/(const matrix<T> &mat, const S &val)
{
    unsigned int elem = mat.elem();
    matrix<T> ret(mat.size());
    T *rptr = ret.ptr();
    const T *mptr = mat.ptr();
    #pragma omp parallel for simd
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = mptr[i] / val;
    }
    return ret;
}

template <typename T, typename S>
matrix<T> operator+(const S &val, const matrix<T> &mat)
{
    return mat + val;
}

template <typename T, typename S>
matrix<T> operator-(const S &val, const matrix<T> &mat)
{
    unsigned int elem = mat.elem();
    matrix<T> ret(mat.size());
    T *rptr = ret.ptr();
    const T *mptr = mat.ptr();
    #pragma omp parallel for simd
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = val - mptr[i];
    }
    return ret;
}

template <typename T, typename S>
matrix<T> operator*(const S &val, const matrix<T> &mat)
{
    return mat * val;
}

template <typename T, typename S>
matrix<T> operator/(const S &val, const matrix<T> &mat)
{
    unsigned int rows = mat.rows(), cols = mat.cols(), elem = mat.elem();
    matrix<T> ret(rows, cols);
    T *rptr = ret.ptr();
    const T *mptr = mat.ptr();
    #pragma omp parallel for simd
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = val / mptr[i];
    }
    return ret;
}

}

#endif
