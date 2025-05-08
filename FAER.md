# FAER.md - Rust Linear Algebra Library Reference

This document provides a concise reference for LLMs when working with the faer linear algebra library.

## Current Versions

- **`faer`**: 0.22.6
- **`faer-core`**: 0.17.1
- **`faer-ext`**: 0.6.0

## Overview

`faer` is a high-performance general-purpose linear algebra library written in Rust, optimized for medium to large dense matrices. It provides efficient implementations of matrix decompositions and algebraic operations with a focus on numerical stability.

```rust
use faer::{mat::*, prelude::*};

// Create a 3x3 matrix
let mut a = Mat::from_fn(3, 3, |i, j| (i + j) as f64);

// Perform Cholesky decomposition
let chol = a.cholesky().unwrap();

// Solve a linear system
let b = vec![1.0, 2.0, 3.0];
let x = chol.solve(&b);
```

## Library Structure

Faer is organized into multiple crates:

1. **`faer`**: High-level crate that re-exports functionality from other crates
2. **`faer-core`**: Low-level building blocks and core matrix operations
3. **`faer-ext`**: Interoperability with other linear algebra libraries (nalgebra, ndarray, etc.)
4. **`faer-svd`**, **`faer-lu`**, etc.: Specialized algorithm implementations

## Cargo Dependencies

```toml
[dependencies]
# High-level API (recommended for most users)
faer = "0.22.6"

# Low-level API (for performance-critical code)
faer-core = "0.17.1"

# Interoperability with other libraries
faer-ext = { version = "0.6.0", features = ["nalgebra", "ndarray"] }
```

## Feature Flags

```toml
[dependencies]
faer = { version = "0.22.6", features = [
    # Default features
    "std",     # Standard library support
    "rayon",   # Parallel computation

    # Optional features
    "serde",   # Serialization/deserialization
    "npy",     # NumPy format support
    "nightly", # Experimental SIMD features
    "perf-warn" # Performance warning diagnostics
]}
```

## Core Modules

### `faer::mat` - Matrix Types

- **`Mat<T>`**: Heap-allocated, resizable matrix (like 2D Vec)
- **`MatRef<'_, T>`**: Immutable matrix view (like 2D slice)
- **`MatMut<'_, T>`**: Mutable matrix view (like mutable 2D slice)

```rust
// Create matrices
let a = Mat::<f64>::zeros(3, 3);  // 3x3 zero matrix
let b = Mat::<f64>::identity(3);  // 3x3 identity matrix
let c = Mat::<f64>::from_fn(3, 3, |i, j| i as f64 + j as f64);  // Custom values

// Access elements
let val = a.read(1, 2);  // Read at row 1, col 2
let mut a_mut = a.as_mut();
a_mut.write(1, 2, 5.0);  // Write at row 1, col 2

// Matrix views
let view = a.as_ref().slice(0..2, 1..3);  // View of a sub-matrix
```

### `faer-core` - Low-level Matrix Operations

The `faer-core` crate provides the fundamental building blocks:

```rust
use faer_core::{mat, mul};

// Create matrix views from raw memory
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let mat_ref = mat::from_column_major_slice(&data, 2, 3);

// Matrix multiplication with fine-grained control
let mut result = vec![0.0; 6];
let mut result_mat = mat::from_column_major_slice_mut(&mut result, 2, 3);
mul::matmul(
    1.0, // alpha
    &mat_ref, // lhs
    &mat_ref, // rhs
    0.0, // beta
    &mut result_mat, // accumulator
);
```

Key modules in `faer-core`:
- **`mul`**: Matrix multiplication operations
- **`solve`**: Triangular solve and other equation solvers
- **`householder`**: Householder transformations for matrix decompositions
- **`mat`**: Low-level matrix view creation utilities

### `faer-ext` - Interoperability with Other Libraries

The `faer-ext` crate enables converting between faer and other linear algebra libraries:

```rust
use faer::Mat;
use faer_ext::*;

// Convert between faer and nalgebra
let faer_mat = Mat::<f32>::identity(8, 7);
let nalgebra_mat = faer_mat.as_ref().into_nalgebra();  // faer -> nalgebra

let nalgebra_mat = nalgebra::DMatrix::<f32>::identity(8, 7);
let faer_mat = nalgebra_mat.view_range(.., ..).into_faer();  // nalgebra -> faer

// Convert between faer and ndarray
let faer_mat = Mat::<f32>::identity(8, 7);
let ndarray_mat = faer_mat.as_ref().into_ndarray();  // faer -> ndarray

let ndarray_mat = ndarray::Array2::<f32>::eye(8, 7);
let faer_mat = ndarray_mat.view().into_faer();  // ndarray -> faer
```

Key traits in `faer-ext`:
- **`IntoFaer`**: Convert from other libraries to faer
- **`IntoNalgebra`**: Convert from faer to nalgebra
- **`IntoNdarray`**: Convert from faer to ndarray

Note: Only matrix views can be converted, not owning matrices.

### `faer::linalg` - Linear Algebra Operations

#### Matrix Decompositions

- **Cholesky**: `matrix.cholesky()`
- **LU**: `matrix.lu()`
- **QR**: `matrix.qr()`
- **SVD**: `matrix.svd()`
- **Eigendecomposition**: `matrix.eigendecomposition()`

#### Memory Management

Faer uses `MemStack` for temporary computational space:

```rust
use faer::linalg::MemoryRequirements;

// Calculate memory requirements for operation
let mem_req = SomeDecomposition::mem_req(rows, cols);
let mut mem = faer::MemStack::new();

// Allocate memory and perform operation
let scratch = mem.get_aligned(mem_req);
let result = SomeDecomposition::compute(matrix, scratch);
```

### `faer::sparse` - Sparse Matrix Support

- **`SparseColMat<T>`**: Column-major sparse matrix
- **`SparseRowMat<T>`**: Row-major sparse matrix

```rust
// Create a sparse matrix
let mut builder = SparseColMatBuilder::<f64>::new(3, 3);
builder.push(0, 0, 1.0);  // row 0, col 0, value 1.0
builder.push(1, 1, 2.0);  // row 1, col 1, value 2.0
builder.push(2, 2, 3.0);  // row 2, col 2, value 3.0
let sparse_matrix = builder.build();
```

## Common Operations

### Matrix Creation

```rust
// From raw data
let data = vec![1.0, 2.0, 3.0, 4.0];
let mat = Mat::from_fn(2, 2, |i, j| data[i * 2 + j]);

// Special matrices
let zeros = Mat::<f64>::zeros(3, 3);
let ones = Mat::<f64>::ones(3, 3);
let identity = Mat::<f64>::identity(3);
let random = Mat::<f64>::from_fn(3, 3, |_, _| rand::random());

// From slices with different memory layouts
use faer_core::mat;
let col_major_mat = mat::from_column_major_slice(&data, 2, 2);
let row_major_mat = mat::from_row_major_slice(&data, 2, 2);
```

### Matrix Operations

```rust
// Arithmetic
let c = &a + &b;  // Addition
let d = &a - &b;  // Subtraction
let e = &a * &b;  // Matrix multiplication
let f = a.scale(2.0);  // Scalar multiplication

// Element-wise operations
let g = a.map(|x| x.sqrt());  // Apply function to each element

// Transposition
let h = a.transpose();

// Low-level matrix multiplication with faer-core
use faer_core::mul;
mul::matmul(
    alpha,  // Scaling factor for result
    &lhs,   // Left matrix
    &rhs,   // Right matrix
    beta,   // Scaling factor for accumulator
    &mut acc // Accumulator matrix
);
```

### Solving Linear Systems

```rust
// Solve Ax = b
let x = a.solve(&b).unwrap();

// With decompositions for repeated solves
let chol = a.cholesky().unwrap();
let x1 = chol.solve(&b1);
let x2 = chol.solve(&b2);

// Triangular solves with faer-core
use faer_core::solve;
solve::solve_triangular_in_place(
    TriangularType::Lower,
    DiagType::NonUnit,
    &triangular_matrix,
    &mut rhs_and_solution
);
```

## Interoperability with Other Libraries

Using the `faer-ext` crate, you can easily convert between faer and other popular libraries:

### Nalgebra Integration

```rust
use faer::Mat;
use faer_ext::*;

// Create matrices in both libraries
let faer_matrix = Mat::<f64>::identity(3, 3);
let nalgebra_matrix = nalgebra::DMatrix::<f64>::identity(3, 3);

// Convert from nalgebra to faer
let converted_to_faer = nalgebra_matrix.view_range(.., ..).into_faer();

// Convert from faer to nalgebra
let converted_to_nalgebra = faer_matrix.as_ref().into_nalgebra();

// Now use with nalgebra's ecosystem
let result = converted_to_nalgebra.transpose() * converted_to_nalgebra;
```

### Ndarray Integration

```rust
use faer::Mat;
use faer_ext::*;

// Create matrices in both libraries
let faer_matrix = Mat::<f64>::identity(3, 3);
let ndarray_matrix = ndarray::Array2::<f64>::eye(3, 3);

// Convert from ndarray to faer
let converted_to_faer = ndarray_matrix.view().into_faer();

// Convert from faer to ndarray
let converted_to_ndarray = faer_matrix.as_ref().into_ndarray();

// Now use with ndarray's ecosystem
let sum = converted_to_ndarray.sum();
```

### Integration Limitations

- Only matrix views can be converted, not owning matrices
- Requires enabling specific features in your Cargo.toml:
  ```toml
  faer-ext = { version = "0.6.0", features = ["nalgebra", "ndarray"] }
  ```

## Performance Optimization

Faer is designed for high-performance linear algebra operations and provides several mechanisms for optimization:

### SIMD Vectorization

Faer uses the `pulp` crate for SIMD (Single Instruction, Multiple Data) operations, which provides:

- Automatic CPU feature detection at runtime
- Support for various instruction sets (SSE, AVX, AVX2, AVX-512)
- Vectorized implementations of core operations

To enable experimental SIMD features:

```toml
# In Cargo.toml
faer = { version = "0.22.6", features = ["nightly"] }
```

Example of SIMD-accelerated matrix addition:

```rust
// Automatically uses the best SIMD instructions available on your CPU
let c = &a + &b;  // Addition using SIMD when possible
```

### Parallelization

Faer leverages the Rayon library for parallel computation:

```rust
use faer_core::Parallelism;

// Sequential execution
let chol1 = matrix.cholesky_in_place(Parallelism::None);

// Parallel execution with Rayon (uses all available threads)
let chol2 = matrix.cholesky_in_place(Parallelism::Rayon(0)); 

// Parallel execution with specific thread count
let chol3 = matrix.cholesky_in_place(Parallelism::Rayon(4)); // Use 4 threads
```

Global parallelization control:

```rust
// Set global parallelism setting
faer::set_global_parallelism(Parallelism::Rayon(8));

// Get current global parallelism setting
let current = faer::get_global_parallelism();

// Temporarily disable parallelism for specific operations
let _guard = faer::disable_global_parallelism();
let result = some_operation();
// Parallelism is automatically restored when _guard is dropped
```

### Memory Optimization Strategies

1. **Matrix Layout**:
   - Use column-major layout for operations on columns
   - Use row-major layout for operations on rows
   - Match layout to access patterns for cache efficiency

2. **Memory Reuse**:
   - Create a memory stack once and reuse it for multiple operations:
   ```rust
   let mut mem = faer::MemStack::new();
   // Reuse for multiple decompositions
   let scratch1 = mem.get_aligned(op1_mem_req);
   let result1 = op1.compute(matrix1, scratch1);
   // Reset and reuse for another operation
   mem.reset();
   let scratch2 = mem.get_aligned(op2_mem_req);
   let result2 = op2.compute(matrix2, scratch2);
   ```

3. **Avoiding Unnecessary Allocations**:
   - Use views (`MatRef`, `MatMut`) rather than creating new matrices
   - Provide preallocated buffers for outputs when possible

### Performance Tips

1. **Decomposition Choice**:
   - For symmetric positive-definite matrices: Cholesky (fastest)
   - For general square matrices: LU
   - For least-squares problems: QR
   - For ill-conditioned matrices: SVD (slowest)

2. **Matrix Size Considerations**:
   - Small matrices (<25x25): Consider nalgebra or cgmath
   - Medium/large matrices: Faer is optimal
   - Use block algorithms for very large matrices

3. **Enable Diagnostic Warnings**:
   - Use the `perf-warn` feature to detect suboptimal usages:
   ```toml
   faer = { version = "0.22.6", features = ["perf-warn"] }
   ```

## Error Handling

Many decompositions return `Result` types to handle numerical issues:

```rust
match matrix.cholesky() {
    Ok(chol) => {
        // Decomposition succeeded
        let solution = chol.solve(&b);
    },
    Err(e) => {
        // Matrix not positive-definite
        println!("Decomposition failed: {}", e);
    }
}
```

## Important Traits

- **`AsMat`**: For owning matrix types
- **`AsMatMut`**: For convertible mutable matrix types
- **`AsMatRef`**: For convertible matrix view types
- **`MatIndex`**: For slicing matrices
- **`IntoFaer`**: For converting external matrix views to faer
- **`IntoNalgebra`**: For converting faer views to nalgebra
- **`IntoNdarray`**: For converting faer views to ndarray

## Advanced Usage with faer-core

For specialized applications requiring maximum control over memory and computation:

```rust
use faer_core::{mat, mul, Parallelism};

// Create matrices from existing memory
let a_data = vec![1.0, 2.0, 3.0, 4.0];
let b_data = vec![5.0, 6.0, 7.0, 8.0];
let mut c_data = vec![0.0; 4];

let a = mat::from_column_major_slice(&a_data, 2, 2);
let b = mat::from_column_major_slice(&b_data, 2, 2);
let mut c = mat::from_column_major_slice_mut(&mut c_data, 2, 2);

// Perform multiplication with specific parallelism strategy
mul::matmul_with_conj(
    1.0,                  // alpha
    false,                // conjugate lhs
    &a,                   // lhs
    false,                // conjugate rhs
    &b,                   // rhs
    0.0,                  // beta
    &mut c,               // accumulator
    Parallelism::Rayon    // parallelism strategy
);
```