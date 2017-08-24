#ifndef MATRIX_LOGI_H_
#define MATRIX_LOGI_H_

namespace acspo {

inline matrix<bool> operator!(const matrix<bool> &mat)
{
    unsigned int elem = mat.elem();
    matrix<bool> ret(mat.size());
    bool *rptr = ret.ptr();
    const bool *mptr = mat.ptr();
    #pragma omp parallel for simd
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = !mptr[i];
    }
    return ret;
}

inline matrix<bool> operator&&(const matrix<bool> &mat1, const matrix<bool> &mat2)
{
    if (mat1.size() != mat2.size()) {
        throw std::runtime_error("dimension mismatch");
    }
    unsigned int elem = mat1.elem();
    matrix<bool> ret(mat1.size());
    bool *rptr = ret.ptr();
    const bool *m1ptr = mat1.ptr();
    const bool *m2ptr = mat2.ptr();
    #pragma omp parallel for simd
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = m1ptr[i] && m2ptr[i];
    }
    return ret;
}

inline matrix<bool> operator||(const matrix<bool> &mat1, const matrix<bool> &mat2)
{
    if (mat1.size() != mat2.size()) {
        throw std::runtime_error("dimension mismatch");
    }
    unsigned int elem = mat1.elem();
    matrix<bool> ret(mat1.size());
    bool *rptr = ret.ptr();
    const bool *m1ptr = mat1.ptr();
    const bool *m2ptr = mat2.ptr();
    #pragma omp parallel for simd
    for (unsigned int i = 0; i < elem; i++) {
        rptr[i] = m1ptr[i] || m2ptr[i];
    }
    return ret;
}

}

#endif
