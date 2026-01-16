# LinAlg

A modern, header-only C++20 linear algebra library designed for performance and ease of use.

## Features

* **Modern C++ Design:** Built using C++20 Concepts (`template <Scalar T>`) and `std::span` for type safety and modern memory management.
* **Matrix Decompositions:**
    * **LU Decomposition:** For matrix inversion and determinant calculation.
    * **QR Decomposition:** Implemented via Gram-Schmidt for eigenvalue algorithms.
* **Eigensolvers:**
    * **Power Iteration:** To find dominant eigenvalues.
    * **QR Algorithm:** For computing full eigenpairs (eigenvalues & eigenvectors).
* **Statistical Tools:** Built-in methods for calculating Mean and Covariance matrices.

## Usage

Since this is a header-only library, simply include the header in your project.

```cpp
#include "linalg/Matrix.hpp"

int main() {
    // Create a 3x3 Identity Matrix
    auto mat = linalg::Matrix<double>::identity(3);

    // Perform operations
    auto inverse = mat.inverse();
    auto eigen_pairs = mat.eigen();
    
    mat.print();
    return 0;
}
