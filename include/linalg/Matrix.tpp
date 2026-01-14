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
    Matrix<T> Matrix<T>::identity(size_type n)
    {
        Matrix<T> result(n, n, 0);
        for (size_type i = 0; i < n; i++)
        {
            result(i, i) = 1;
        }
        return result;
    }

    template <Scalar T>
    void Matrix<T>::print() const
    {
        std::cout << "[" << rows_ << "x" << cols_ << " Matrix]:\n";
        for (size_type i = 0; i < rows_; ++i)
        {
            for (size_type j = 0; j < cols_; ++j)
            {
                T val = (*this)(i, j);
                if (std::abs(val) < 1e-10)
                    val = 0.0;

                std::cout << std::setw(8) << std::setprecision(4) << val << " ";
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

    template <Scalar T>
    auto Matrix<T>::lu_decompose() const -> std::pair<Matrix, std::vector<size_type>>
    {
        if (rows_ != cols_)
            throw std::invalid_argument("LU requires square matrix");

        size_type n = rows_;
        Matrix result = *this;
        std::vector<size_type> P(n);

        for (size_type i = 0; i < n; ++i)
            P[i] = i;

        for (size_type i = 0; i < n; ++i)
        {
            T maxVal = 0;
            size_type pivotRow = i;
            for (size_type k = i; k < n; ++k)
            {
                if (std::abs(result(k, i)) > maxVal)
                {
                    maxVal = std::abs(result(k, i));
                    pivotRow = k;
                }
            }

            if (maxVal < 1e-9)
                throw std::runtime_error("Matrix is singular (Cannot invert)");

            if (pivotRow != i)
            {
                std::swap(P[i], P[pivotRow]);
                for (size_type j = 0; j < n; ++j)
                {
                    std::swap(result(i, j), result(pivotRow, j));
                }
            }

            for (size_type j = i + 1; j < n; ++j)
            {
                result(j, i) /= result(i, i);
                for (size_type k = i + 1; k < n; ++k)
                {
                    result(j, k) -= result(j, i) * result(i, k);
                }
            }
        }

        return {result, P};
    }

    template <Scalar T>
    T Matrix<T>::determinant() const
    {
        try
        {
            auto [lu, P] = lu_decompose();
            T det = 1;
            for (size_type i = 0; i < rows_; ++i)
            {
                det *= lu(i, i);
            }

            int swaps = 0;
            for (size_type i = 0; i < rows_; ++i)
                if (P[i] != i)
                    swaps++;
            if ((swaps % 2) != 0)
                det = -det;

            return det;
        }
        catch (...)
        {
            return 0;
        }
    }

    template <Scalar T>
    Matrix<T> Matrix<T>::inverse() const
    {
        auto [lu, P] = lu_decompose();
        size_type n = rows_;
        Matrix inv(n, n);

        for (size_type j = 0; j < n; ++j)
        {

            std::vector<T> x(n, 0);
            for (size_type i = 0; i < n; ++i)
            {
                if (P[i] == j)
                    x[i] = 1;
            }

            for (size_type i = 0; i < n; ++i)
            {
                for (size_type k = 0; k < i; ++k)
                {
                    x[i] -= lu(i, k) * x[k];
                }
            }

            for (int i = n - 1; i >= 0; --i)
            {
                for (size_type k = i + 1; k < n; ++k)
                {
                    x[i] -= lu(i, k) * x[k];
                }
                x[i] /= lu(i, i);
            }

            for (size_type i = 0; i < n; ++i)
            {
                inv(i, j) = x[i];
            }
        }
        return inv;
    }

    template <Scalar T>
    Matrix<T> Matrix<T>::transpose() const
    {
        Matrix<T> result(cols_, rows_);

        const size_type blockSize = 64;

        for (size_type i = 0; i < rows_; i += blockSize)
        {
            for (size_type j = 0; j < cols_; j += blockSize)
            {

                size_type iMax = std::min(i + blockSize, rows_);
                size_type jMax = std::min(j + blockSize, cols_);

                for (size_type ii = i; ii < iMax; ++ii)
                {
                    for (size_type jj = j; jj < jMax; ++jj)
                    {
                        result(jj, ii) = (*this)(ii, jj);
                    }
                }
            }
        }
        return result;
    }

    template <Scalar T>
    T Matrix<T>::column_norm(size_type col_idx) const
    {
        T sum = 0;
        for (size_type i = 0; i < rows_; ++i)
        {
            sum += (*this)(i, col_idx) * (*this)(i, col_idx);
        }
        return std::sqrt(sum);
    }

    template <Scalar T>
    std::pair<Matrix<T>, Matrix<T>> Matrix<T>::qr_decompose() const
    {
        size_type n = rows_;
        size_type m = cols_;

        Matrix<T> Q(n, m);
        Matrix<T> R(m, m, 0);

        Q = *this;

        for (size_type i = 0; i < m; ++i)
        {
            R(i, i) = Q.column_norm(i);

            for (size_type k = 0; k < n; ++k)
            {
                Q(k, i) /= R(i, i);
            }

            for (size_type j = i + 1; j < m; ++j)
            {
                T dot = 0;
                for (size_type k = 0; k < n; ++k)
                {
                    dot += Q(k, i) * Q(k, j);
                }
                R(i, j) = dot;

                for (size_type k = 0; k < n; ++k)
                {
                    Q(k, j) -= dot * Q(k, i);
                }
            }
        }

        return {Q, R};
    }

    template <Scalar T>
    auto Matrix<T>::eigen() const -> EigenPairs
    {
        if (rows_ != cols_)
            throw std::invalid_argument("Matrix must be square");

        size_type n = rows_;
        Matrix<T> A_iter = *this;

        size_type max_iter = 1000;
        for (size_type k = 0; k < max_iter; ++k)
        {
            T off_diagonal_sum = 0;
            for (size_type i = 1; i < n; ++i)
            {
                for (size_type j = 0; j < i; ++j)
                {
                    off_diagonal_sum += std::abs(A_iter(i, j));
                }
            }
            if (off_diagonal_sum < 1e-9)
                break;

            auto [Q, R] = A_iter.qr_decompose();
            A_iter = R * Q;
        }

        std::vector<T> eigenvalues;
        for (size_type i = 0; i < n; ++i)
        {
            eigenvalues.push_back(A_iter(i, i));
        }

        std::vector<Matrix<T>> eigenvectors;
        Matrix<T> I = Matrix<T>::identity(n);

        for (T lambda : eigenvalues)
        {
            Matrix<T> v(n, 1, T{1});

            T perturbed_lambda = lambda + 1e-6;

            Matrix<T> M = *this;
            for (size_type i = 0; i < n; ++i)
            {
                M(i, i) -= perturbed_lambda;
            }

            Matrix<T> M_inv = M.inverse();

            for (int iter = 0; iter < 10; ++iter)
            {
                v = M_inv * v;

                T norm = v.column_norm(0);
                for (size_type i = 0; i < n; ++i)
                    v(i, 0) /= norm;
            }
            eigenvectors.push_back(v);
        }

        return {eigenvalues, eigenvectors};
    }

    template <Scalar T>
    auto Matrix<T>::power_iteration(size_t k) const -> EigenPairs
    {
        if (rows_ != cols_)
            throw std::invalid_argument("Matrix must be square");

        size_type n = rows_;
        Matrix<T> A = *this;

        std::vector<T> eigenvalues;
        std::vector<Matrix<T>> eigenvectors;

        for (size_t i = 0; i < k; ++i)
        {

            Matrix<T> v(n, 1, T{1});

            T prev_eigenvalue = 0;
            T eigenvalue = 0;

            for (int iter = 0; iter < 1000; ++iter)
            {
                // v = A * v
                Matrix<T> v_next(n, 1, 0);

                for (size_type r = 0; r < n; ++r)
                {
                    T sum = 0;
                    for (size_type c = 0; c < n; ++c)
                    {
                        sum += A(r, c) * v(c, 0);
                    }
                    v_next(r, 0) = sum;
                }
                v = v_next;

                T norm = v.column_norm(0);
                if (norm < 1e-9)
                    break; // Safety
                for (size_type r = 0; r < n; ++r)
                    v(r, 0) /= norm;

                eigenvalue = norm;

                if (std::abs(eigenvalue - prev_eigenvalue) < 1e-6)
                {
                    break;
                }
                prev_eigenvalue = eigenvalue;
            }

            eigenvalues.push_back(eigenvalue);
            eigenvectors.push_back(v);

            for (size_type r = 0; r < n; ++r)
            {
                for (size_type c = 0; c < n; ++c)
                {
                    A(r, c) -= eigenvalue * v(r, 0) * v(c, 0);
                }
            }
        }

        return {eigenvalues, eigenvectors};
    }

    template <Scalar T>
    Matrix<T> Matrix<T>::mean(int axis) const
    {
        // axis=0 means "collapse rows" (calculate mean for each column)
        if (axis != 0)
            throw std::invalid_argument("Only axis=0 supported for now");

        Matrix<T> result(1, cols_, 0);

        for (size_type i = 0; i < rows_; ++i)
        {
            for (size_type j = 0; j < cols_; ++j)
            {
                result(0, j) += (*this)(i, j);
            }
        }

        for (size_type j = 0; j < cols_; ++j)
        {
            result(0, j) /= static_cast<T>(rows_);
        }

        return result;
    }

    template <Scalar T>
    Matrix<T> Matrix<T>::covariance() const
    {
        if (rows_ < 2)
            throw std::runtime_error("Need at least 2 rows for covariance");

        Matrix<T> cov(cols_, cols_);
        T scale = T{1} / (rows_ - 1);

        for (size_type i = 0; i < cols_; ++i)
        {
            for (size_type j = i; j < cols_; ++j)
            {
                T sum = 0;
                for (size_type k = 0; k < rows_; ++k)
                {
                    sum += (*this)(k, i) * (*this)(k, j);
                }

                T val = sum * scale;
                cov(i, j) = val;

                if (i != j)
                    cov(j, i) = val;
            }
        }
        return cov;
    }
}