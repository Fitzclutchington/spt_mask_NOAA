#ifndef MATRIX_STAT_H_
#define MATRIX_STAT_H_

#include <vector>
#include <algorithm>

namespace acspo {

template <typename T>
double sum(const matrix<T> &mat)
{
    unsigned int elem = mat.elem();
    unsigned int count = 0;
    double ret = 0;
    #pragma omp parallel for reduction(+:ret,count)
    for (unsigned int i = 0; i < elem; i++) {
        if (!std::isnan(mat(i))) {
            ret += mat(i);
            count++;
        }
    }
    if (count == 0) {
        return NAN;
    }
    return ret;
}

template <typename T>
double mean(const matrix<T> &mat)
{
    unsigned int elem = mat.elem();
    unsigned int count = 0;
    double ret = 0;
    #pragma omp parallel for reduction(+:ret,count)
    for (unsigned int i = 0; i < elem; i++) {
        if (!std::isnan(mat(i))) {
            ret += mat(i);
            count++;
        }
    }
    if (count == 0) {
        return NAN;
    }
    ret /= count;
    return ret;
}

template <typename T>
double var(const matrix<T> &mat, double avg)
{
    unsigned int elem = mat.elem();
    unsigned int count = 0;
    double ret = 0;
    #pragma omp parallel for reduction(+:ret,count)
    for (unsigned int i = 0; i < elem; i++) {
        if (!std::isnan(mat(i))) {
            ret += (mat(i)-avg)*(mat(i)-avg);
            count++;
        }
    }
    if (count == 0) {
        return NAN;
    }
    ret /= count;
    return ret;
}

template <typename T>
double var(const matrix<T> &mat)
{
    return var(mat, mean(mat));
}

template <typename T>
double std_dev(const matrix<T> &mat, double avg)
{
    return std::sqrt(var(mat, avg));
}

template <typename T>
double std_dev(const matrix<T> &mat) 
{
    return std::sqrt(var(mat));
}

template <typename T>
double med(const matrix<T> &mat)
{
    unsigned int elem = mat.elem();
    std::vector<double> buf;
    buf.reserve(elem);
    for (unsigned int i = 0; i < elem; i++) {
        if (!std::isnan(mat(i))) {
            buf.push_back(mat(i));
        }
    }
    if (buf.size() == 0) {
        return NAN;
    }
    std::sort(buf.begin(), buf.end());
    if (buf.size() % 2 == 1) {
        return buf[(buf.size()-1)/2];
    } else {
        return 0.5*(buf[buf.size()/2-1]+buf[buf.size()/2]);
    }
}

}

#endif
