#pragma once

#include <vector>
#include <concepts>
#include <type_traits>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <span>
#include <cmath>

namespace linalg
{

    template <typename T>
    concept Scalar = std::is_arithmetic_v<T>;

    template <Scalar T>
    class Matrix
    {
    public:
        using value_type = T;
        using size_type = std::size_t;

        explicit Matrix(size_type rows, size_type cols, T initial_value = T{});

        Matrix(const Matrix &) = default;
        Matrix &operator=(const Matrix &) = default;
        Matrix(Matrix &&) noexcept = default;
        Matrix &operator=(Matrix &&) noexcept = default;
        ~Matrix() = default;

        std::span<T> operator[](size_type row)
        {
            return std::span<T>(&data_[row * cols_], cols_);
        }

        std::span<const T> operator[](size_type row) const
        {
            return std::span<const T>(&data_[row * cols_], cols_);
        }

        T &operator()(size_type row, size_type col);
        const T &operator()(size_type row, size_type col) const;

        [[nodiscard]] size_type rows() const noexcept;
        [[nodiscard]] size_type cols() const noexcept;
        [[nodiscard]] size_type size() const noexcept;

        [[nodiscard]] static Matrix identity(size_type n);

        void print() const;

        Matrix &operator+=(const Matrix &other);
        [[nodiscard]] Matrix operator+(const Matrix &other) const;

        Matrix &operator-=(const Matrix &other);
        [[nodiscard]] Matrix operator-(const Matrix &other) const;

        [[nodiscard]] Matrix operator*(const Matrix &other) const;

        [[nodiscard]] T determinant() const;

        [[nodiscard]] Matrix inverse() const;

        [[nodiscard]] Matrix transpose() const;

        struct EigenPairs
        {
            std::vector<T> eigenvalues;
            std::vector<Matrix<T>> eigenvectors;
        };

        [[nodiscard]] EigenPairs eigen() const;

        [[nodiscard]] EigenPairs power_iteration(size_type k) const;

        [[nodiscard]] Matrix mean(int axis = 0) const;

        [[nodiscard]] Matrix covariance() const;

    private:
        std::vector<T> data_;
        size_type rows_;
        size_type cols_;

        [[nodiscard]] size_type index(size_type r, size_type c) const;

        [[nodiscard]] std::pair<Matrix, std::vector<size_type>> lu_decompose() const;

        [[nodiscard]] std::pair<Matrix, Matrix> qr_decompose() const;

        [[nodiscard]] T column_norm(size_type col_idx) const;
    };

}

#include "Matrix.tpp"