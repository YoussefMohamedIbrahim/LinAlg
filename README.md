# LinAlg

A lightweight, header-only linear algebra library written in modern C++20.

`LinAlg` provides a generic `linalg::Matrix<T>` type for arithmetic scalar values (`int`, `float`, `double`, ...), along with core matrix operations and practical numerical routines.


## Features

- Header-only design (`include/linalg/Matrix.hpp`, `include/linalg/Matrix.tpp`)
- CMake `INTERFACE` library target (`LinAlg`)
- C++20 concept-based scalar constraint (`Scalar`)
- Bounds-checked indexed access
- Matrix arithmetic: addition, subtraction, multiplication
- Linear algebra routines:
  - identity matrix creation
  - transpose
  - determinant (LU-based)
  - inverse (LU solve)
  - eigen decomposition
  - top-k eigenpairs via power iteration + deflation
- Data helpers:
  - column-wise mean (`axis = 0`)
  - covariance matrix

## Requirements

- CMake 3.14+
- C++20 compiler
  - GCC 11+ (recommended)
  - Clang 14+ (recommended)
  - Recent MSVC with C++20 support

## Repository Structure

```text
LinAlg/
├── CMakeLists.txt
├── include/
│   └── linalg/
│       ├── Matrix.hpp
│       └── Matrix.tpp
└── build/            # optional local build directory
```

## Build and Install

Even though this is header-only, install/export rules are provided for clean CMake integration.

```bash
cmake -S . -B build
cmake --build build
cmake --install build --prefix ./install
```

Install result:

- Headers: `install/include/linalg`
- CMake package config export: `install/lib/cmake/LinAlg/LinAlgTargets.cmake`

## How to Use in Your Project

### Option A: `add_subdirectory`

```cmake
cmake_minimum_required(VERSION 3.14)
project(MyApp LANGUAGES CXX)

add_subdirectory(path/to/LinAlg)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE LinAlg)
target_compile_features(my_app PRIVATE cxx_std_20)
```

### Option B: `find_package`

```cmake
cmake_minimum_required(VERSION 3.14)
project(MyApp LANGUAGES CXX)

find_package(LinAlg REQUIRED CONFIG)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE LinAlg::LinAlg)
target_compile_features(my_app PRIVATE cxx_std_20)
```

Configure your app with LinAlg install prefix:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/LinAlg/install
```

## Quick Start

```cpp
#include <iostream>
#include <linalg/Matrix.hpp>

int main() {
    using linalg::Matrix;

    Matrix<double> A(2, 2, 0.0);
    A(0, 0) = 1.0; A(0, 1) = 2.0;
    A(1, 0) = 3.0; A(1, 1) = 4.0;

    Matrix<double> I = Matrix<double>::identity(2);
    Matrix<double> C = A + I;
    Matrix<double> P = A * C;

    std::cout << "det(A) = " << A.determinant() << "\n";
    P.print();

    return 0;
}
```

## API Reference

Namespace: `linalg`

### Concepts

- `template <typename T> concept Scalar`
  - Accepts arithmetic scalar types.

### Class

- `template <Scalar T> class Matrix`

#### Types

- `value_type`
- `size_type`

#### Constructors / Assignment

- `Matrix(size_type rows, size_type cols, T initial_value = T{})`
- `Matrix()`
- Copy/move ctor and assignment are defaulted.

#### Element Access

- `T& operator()(size_type row, size_type col)`
- `const T& operator()(size_type row, size_type col) const`
- `std::span<T> operator[](size_type row)`
- `std::span<const T> operator[](size_type row) const`

#### Shape

- `size_type rows() const noexcept`
- `size_type cols() const noexcept`
- `size_type size() const noexcept`

#### Factory / Utility

- `static Matrix identity(size_type n)`
- `void print() const`

#### Arithmetic

- `Matrix& operator+=(const Matrix& other)`
- `Matrix operator+(const Matrix& other) const`
- `Matrix& operator-=(const Matrix& other)`
- `Matrix operator-(const Matrix& other) const`
- `Matrix operator*(const Matrix& other) const`

#### Linear Algebra

- `T determinant() const`
- `Matrix inverse() const`
- `Matrix transpose() const`
- `EigenPairs eigen() const`
- `EigenPairs power_iteration(size_type k) const`

#### Statistics

- `Matrix mean(int axis = 0) const`
- `Matrix covariance() const`

### EigenPairs

```cpp
struct EigenPairs {
    std::vector<T> eigenvalues;
    std::vector<Matrix<T>> eigenvectors;
};
```

## Examples

### Matrix Arithmetic

```cpp
linalg::Matrix<double> A(2, 2, 0.0);
linalg::Matrix<double> B(2, 2, 0.0);

A(0,0)=1; A(0,1)=2;
A(1,0)=3; A(1,1)=4;

B(0,0)=5; B(0,1)=6;
B(1,0)=7; B(1,1)=8;

auto sum  = A + B;
auto diff = A - B;
auto prod = A * B;
```

### Determinant and Inverse

```cpp
linalg::Matrix<double> M(3, 3, 0.0);
M(0,0)=4; M(0,1)=7; M(0,2)=2;
M(1,0)=3; M(1,1)=6; M(1,2)=1;
M(2,0)=2; M(2,1)=5; M(2,2)=1;

double det = M.determinant();
auto inv = M.inverse();
```

### Eigen Decomposition

```cpp
linalg::Matrix<double> S(2, 2, 0.0);
S(0,0)=2; S(0,1)=1;
S(1,0)=1; S(1,1)=2;

auto eig = S.eigen();
for (std::size_t i = 0; i < eig.eigenvalues.size(); ++i) {
    std::cout << "lambda[" << i << "] = " << eig.eigenvalues[i] << "\n";
    eig.eigenvectors[i].print();
}
```

### Power Iteration (Top-k)

```cpp
auto top2 = S.power_iteration(2);
```

### Mean and Covariance

```cpp
linalg::Matrix<double> X(4, 2, 0.0);
X(0,0)=1; X(0,1)=2;
X(1,0)=2; X(1,1)=3;
X(2,0)=3; X(2,1)=4;
X(3,0)=4; X(3,1)=5;

auto mu  = X.mean(0);      // 1 x features
auto cov = X.covariance(); // features x features
```

## Error Handling

The library uses standard exceptions:

- `std::out_of_range`
  - out-of-bounds indexed access
- `std::invalid_argument`
  - incompatible dimensions in arithmetic
  - non-square matrix for square-only operations
  - unsupported `mean(axis)` values (`axis != 0`)
- `std::runtime_error`
  - singular matrix in LU/inverse path
  - covariance requires at least 2 rows

## Numerical Notes

- Multiplication and transpose use block-based loops for better cache behavior.
- `determinant()` returns `0` if internal LU decomposition fails.
- `covariance()` currently computes scaled second moments (`XᵀX/(n-1)` style on current matrix values); center data first if you need strictly centered covariance.

