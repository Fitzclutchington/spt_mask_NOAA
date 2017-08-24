#ifndef MATRIX_MATH_H_
#define MATRIX_MATH_H_

namespace acspo {

template <typename T>
matrix<T> unary_op(const matrix<T> &mat, T (*op)(T))
{
    unsigned int elem = mat.elem();
    matrix<T> ret(mat.size());
    T *rptr = ret.ptr();
    const T *mptr = mat.ptr();
    #pragma omp parallel for
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = op(mptr[i]);
    }
    return ret;
}

template <typename T>
matrix<T> binary_op(const matrix<T> &mat1, const matrix<T> &mat2, T (*op)(T, T))
{
    if (mat1.size() != mat2.size()) {
        throw std::runtime_error("dimension mismatch");
    }
    unsigned int elem = mat1.elem();
    matrix<T> ret(mat1.size());
    T *rptr = ret.ptr();
    const T *m1ptr = mat1.ptr();
    const T *m2ptr = mat2.ptr();
    #pragma omp parallel for
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = op(m1ptr[i], m2ptr[i]);
    }
    return ret;
}

template <typename T>
matrix<T> abs(const matrix<T> &mat)
{
    return unary_op(mat, std::abs);
}

template <typename T>
matrix<T> sqrt(const matrix<T> &mat)
{
    return unary_op(mat, std::sqrt);
}

template <typename T>
matrix<T> hypot(const matrix<T> &mat1, const matrix<T> &mat2)
{
    return binary_op(mat1, mat2, std::hypot);
}

}

#endif
