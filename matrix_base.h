#ifndef MATRIX_BASE_H_
#define MATRIX_BASE_H_

namespace acspo {

template <typename T>
class matrix {
private:
    std::shared_ptr<T> _refs;
    T *_data;
    unsigned int _rows, _cols, _elem;

public:
    matrix();
    matrix(unsigned int rows, unsigned int cols);
    matrix(std::pair<unsigned int, unsigned int> size);
    matrix(unsigned int rows, unsigned int cols, T *data);
    matrix(std::pair<unsigned int, unsigned int> size, T *_data);
    unsigned int rows() const;
    unsigned int cols() const;
    std::pair<unsigned int, unsigned int> size() const;
    unsigned int elem() const;
    T & at(unsigned int i, unsigned int j);
    T & operator()(unsigned int i, unsigned int j);
    const T & at(unsigned int i, unsigned int j) const;
    const T & operator()(unsigned int i, unsigned int j) const;
    T & at(std::pair<unsigned int, unsigned int> ij);
    T & operator()(std::pair<unsigned int, unsigned int> ij);
    const T & at(std::pair<unsigned int, unsigned int>  ij) const;
    const T & operator()(std::pair<unsigned int, unsigned int> ij) const;
    T & at(unsigned int i);
    T & operator()(unsigned int i);
    const T & at(unsigned int i) const;
    const T & operator()(unsigned int i) const;
    T * ptr(unsigned int i = 0);
    T * operator[](unsigned int i);
    const T * ptr(unsigned int i = 0) const;
    const T * operator[](unsigned int i) const;
    matrix & copy(const matrix &mat);
    matrix & assign(const T &val);
    matrix & assign(const T &val, const matrix<bool> &mask);
    matrix & assign(const T &val1, const T &val2, const matrix<bool> &mask);
    matrix & create(unsigned int rows, unsigned int cols);
    matrix & create(std::pair<unsigned int, unsigned int> size);
    matrix clone() const;
    static matrix zeros(unsigned int rows, unsigned int cols);
    static matrix zeros(std::pair<unsigned int, unsigned int> size);
    template <typename S>
    matrix<S> convert() const;
    void write(const std::string &name, bool newline = true) const;
    void write(std::ostream &os, bool newline = true) const;
};

template <typename T>
matrix<T>::matrix()
{
    _rows = 0;
    _cols = 0;
    _elem = 0;
}

template <typename T>
matrix<T>::matrix(unsigned int rows, unsigned int cols)
{
    _rows = 0;
    _cols = 0;
    _elem = 0;
    create(rows, cols);
}

template <typename T>
matrix<T>::matrix(std::pair<unsigned int, unsigned int> size)
{
    _rows = 0;
    _cols = 0;
    _elem = 0;
    create(size);
}

template <typename T>
matrix<T>::matrix(unsigned int rows, unsigned int cols, T *data)
{
    _rows = rows;
    _cols = cols;
    _elem = rows*cols;
    _data = data;
}

template <typename T>
matrix<T>::matrix(std::pair<unsigned int, unsigned int> size, T *data)
{
    _rows = size.first;
    _cols = size.second;
    _elem = size.first*size.second;
    _data = data;
}

template <typename T>
unsigned int matrix<T>::rows() const
{
    return _rows;
}

template <typename T>
unsigned int matrix<T>::cols() const
{
    return _cols;
}

template <typename T>
std::pair<unsigned int, unsigned int> matrix<T>::size() const
{
    return std::make_pair(_rows, _cols);
}

template <typename T>
unsigned int matrix<T>::elem() const
{
    return _elem;
}

template <typename T>
T & matrix<T>::at(unsigned int i, unsigned int j)
{
    return const_cast<T &>(static_cast<const matrix<T> &>(*this).at(i, j));
}

template <typename T>
T & matrix<T>::operator()(unsigned int i, unsigned int j)
{
    return at(i, j);
}

template <typename T>
T & matrix<T>::at(std::pair<unsigned int, unsigned int> ij)
{
    return at(ij.first, ij.second);
}

template <typename T>
T & matrix<T>::operator()(std::pair<unsigned int, unsigned int> ij)
{
    return at(ij.first, ij.second);
}

template <typename T>
T & matrix<T>::at(unsigned int i)
{
    return const_cast<T &>(static_cast<const matrix<T> &>(*this).at(i));
}

template <typename T>
T & matrix<T>::operator()(unsigned int i)
{
    return at(i);
}

template <typename T>
const T & matrix<T>::at(unsigned int i, unsigned int j) const
{
#ifndef NDEBUG
    if (i >= _rows) {
        throw std::out_of_range("row is out of range");
    }
    if (j >= _cols) {
        throw std::out_of_range("column is out of range");
    }
#endif
    return _data[i*_cols+j];
}

template <typename T>
const T & matrix<T>::operator()(unsigned int i, unsigned int j) const
{
    return at(i, j);
}

template <typename T>
const T & matrix<T>::at(std::pair<unsigned int, unsigned int> ij) const
{
    return at(ij.first, ij.second);
}

template <typename T>
const T & matrix<T>::operator()(std::pair<unsigned int, unsigned int> ij) const
{
    return at(ij.first, ij.second);
}

template <typename T>
const T & matrix<T>::at(unsigned int i) const
{
#ifndef NDEBUG
    if (i >= _elem) {
        throw std::out_of_range("element is out of range");
    }
#endif
    return _data[i];
}

template <typename T>
const T & matrix<T>::operator()(unsigned int i) const
{
    return at(i);
}

template <typename T>
T * matrix<T>::ptr(unsigned int i)
{
    return const_cast<T *>(static_cast<const matrix<T> &>(*this).ptr(i));
}

template <typename T>
T * matrix<T>::operator[](unsigned int i)
{
    return ptr(i);
}

template <typename T>
const T * matrix<T>::ptr(unsigned int i) const
{
#ifndef NDEBUG
    if (i >= _rows) {
        throw std::out_of_range("row is out of range");
    }
#endif
    return &_data[i*_cols];
}

template <typename T>
const T * matrix<T>::operator[](unsigned int i) const
{
    return ptr(i);
}

template <typename T>
matrix<T> & matrix<T>::copy(const matrix &mat)
{
    if (size() != mat.size()) {
        throw std::invalid_argument("dimension mismatch");
    }
    #pragma omp parallel for
    for (int i = 0; i < _elem; i++) {
        _data[i] = mat._data[i];
    }
    return *this;
}

template <typename T>
matrix<T> matrix<T>::clone() const
{
    return matrix(_rows, _cols).copy(*this);
}

template <typename T>
template <typename S>
matrix<S> matrix<T>::convert() const
{
    matrix<S> ret(_rows, _cols);
    S *ptr = ret.ptr();
    #pragma omp parallel for
    for (unsigned int i = 0; i < _elem; i++) {
        ptr[i] = _data[i];
    }
    return ret;
}

template <typename T>
void matrix<T>::write(std::ostream &os, bool newline) const
{
    for (unsigned int i = 0; i < _rows; i++) {
        for (unsigned int j = 0; j < _cols; j++) {
            os << at(i, j);
            if (j < _cols-1) {
                os << " ";
            }
        }
        if (i < _rows-1) {
            os << std::endl;
        }
    }
    if (newline) {
        os << std::endl;
    }
}

template <typename T>
void matrix<T>::write(const std::string &name, bool newline) const
{
    std::ofstream os(name);
    write(os, newline);
}

template <typename T>
std::ostream & operator<<(std::ostream &os, const matrix<T> &mat)
{
    os << mat.rows() << " x " << mat.cols() << " matrix:" << std::endl;
    mat.write(os, false);
    return os;
}

template <typename T>
matrix<T> & matrix<T>::assign(const T &val)
{
    #pragma omp parallel for
    for (unsigned int i = 0; i < _elem; i++) {
        _data[i] = val;
    }
    return *this;
}

template <typename T>
matrix<T> & matrix<T>::assign(const T &val, const matrix<bool> &mask)
{
    if (size() != mask.size()) {
        throw std::invalid_argument("dimension mismatch");
    }
    const bool *ptr = mask.ptr();
    #pragma omp parallel for
    for (unsigned int i = 0; i < _elem; i++) {
        if (ptr[i]) {
            _data[i] = val;
        }
    }
    return *this;
}

template <typename T>
matrix<T> & matrix<T>::assign(const T &val1, const T &val2, const matrix<bool> &mask)
{
    if (size() != mask.size()) {
        throw std::invalid_argument("dimension mismatch");
    }
    const bool *ptr = mask.ptr();
    #pragma omp parallel for
    for (unsigned int i = 0; i < _elem; i++) {
        if (ptr[i]) {
            _data[i] = val1;
        } else {
            _data[i] = val2;
        }
    }
    return *this;
}

template <typename T>
matrix<T> & matrix<T>::create(unsigned int rows, unsigned int cols)
{
    unsigned int elem = rows*cols;

    if (elem == _elem) {
        _rows = rows;
        _cols = cols;
        return *this;
    }

    _rows = rows;
    _cols = cols;
    _elem = elem;
    _refs.reset();
    _data = new T[_elem];
    _refs.reset(_data, std::default_delete<T[]>());
    return *this;
}

template <typename T>
matrix<T> & matrix<T>::create(std::pair<unsigned int, unsigned int> size)
{
    return create(size.first, size.second);
}

template <typename T>
matrix<T> matrix<T>::zeros(unsigned int rows, unsigned int cols)
{
    return matrix(rows, cols).assign(0);
}

template <typename T>
matrix<T> matrix<T>::zeros(std::pair<unsigned int, unsigned int> size)
{
    return matrix(size).assign(0);
}

}

#endif
