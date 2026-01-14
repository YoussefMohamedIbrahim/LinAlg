#ifdef __INTELLISENSE__
#include "Matrix.hpp"
#endif

namespace linalg
{
    template <Scalar T>
    Matrix<T>::Matrix(size_type rows, size_type cols, T initial_value)
        : rows_(rows), cols_(cols)
    {
        if (rows == 0 || cols == 0)
        {
            throw std::invalid_argument("Matrix dimensions must be > 0");
        }
        data_.resize(rows * cols, initial_value);
    }

    template <Scalar T>
    auto Matrix<T>::index(size_type r, size_type c) const -> size_type
    {
        return r * cols_ + c;
    }

    template <Scalar T>
    T &Matrix<T>::operator()(size_type row, size_type col)
    {
        if (row >= rows_ || col >= cols_)
        {
            throw std::out_of_range("Matrix index out of bounds");
        }
        return data_[index(row, col)];
    }

    template <Scalar T>
    const T &Matrix<T>::operator()(size_type row, size_type col) const
    {
        if (row >= rows_ || col >= cols_)
        {
            throw std::out_of_range("Matrix index out of bounds");
        }
        return data_[index(row, col)];
    }

    template <Scalar T>
    auto Matrix<T>::rows() const noexcept -> size_type { return rows_; }

    template <Scalar T>
    auto Matrix<T>::cols() const noexcept -> size_type { return cols_; }

    template <Scalar T>
    auto Matrix<T>::size() const noexcept -> size_type { return data_.size(); }

    template <Scalar T>
    void Matrix<T>::print() const
    {
        std::cout << "[" << rows_ << "x" << cols_ << " Matrix]:\n";
        for (size_type i = 0; i < rows_; ++i)
        {
            for (size_type j = 0; j < cols_; ++j)
            {
                std::cout << std::setw(8) << std::setprecision(4) << (*this)(i, j) << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    template <Scalar T>
    Matrix<T> &Matrix<T>::operator+=(const Matrix &other)
    {
        if (rows_ != other.rows() || cols_ != other.cols())
        {
            throw std::invalid_argument("Dimension mismatch: Cannot add matrices of different sizes.");
        }
        for (size_type i = 0; i < data_.size(); ++i)
        {
            data_[i] += other.data_[i];
        }
        return *this;
    }

    template <Scalar T>
    Matrix<T> Matrix<T>::operator+(const Matrix &other) const
    {
        Matrix<T> result = *this;
        result += other;
        return result;
    }

    template <Scalar T>
    Matrix<T> &Matrix<T>::operator-=(const Matrix &other)
    {
        if (rows_ != other.rows() || cols_ != other.cols())
        {
            throw std::invalid_argument("Dimension mismatch: Cannot subtract matrices of different sizes.");
        }

        for (size_type i = 0; i < data_.size(); ++i)
        {
            data_[i] -= other.data_[i];
        }
        return *this;
    }

    template <Scalar T>
    Matrix<T> Matrix<T>::operator-(const Matrix &other) const
    {
        Matrix<T> result = *this;
        result -= other;
        return result;
    }

    template <Scalar T>
    Matrix<T> Matrix<T>::operator*(const Matrix &other) const
    {
        if (cols_ != other.rows())
        {
            throw std::invalid_argument("Dimension mismatch");
        }

        Matrix<T> result(rows_, other.cols());

        const size_type blockSize = 64;

        for (size_type ii = 0; ii < rows_; ii += blockSize)
        {
            for (size_type kk = 0; kk < cols_; kk += blockSize)
            {
                for (size_type jj = 0; jj < other.cols(); jj += blockSize)
                {
                    size_type iMax = std::min(ii + blockSize, rows_);
                    size_type kMax = std::min(kk + blockSize, cols_);
                    size_type jMax = std::min(jj + blockSize, other.cols());

                    for (size_type i = ii; i < iMax; ++i)
                    {
                        for (size_type k = kk; k < kMax; ++k)
                        {

                            T temp = (*this)(i, k);

                            for (size_type j = jj; j < jMax; ++j)
                            {
                                result[i][j] += temp * other(k, j);
                            }
                        }
                    }
                }
            }
        }
        return result;
    }
}