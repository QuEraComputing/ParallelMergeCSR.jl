# ParallelMergeCSR.jl

[![Build Status](https://github.com/QuEraComputing/Bloqade.jl/workflows/CI/badge.svg)](https://github.com/QuEraComputing/Bloqade.jl/actions)
[![codecov](https://codecov.io/gh/QuEraComputing/ParallelMergeCSR.jl/branch/main/graph/badge.svg?token=P0UCC5CAVB)](https://codecov.io/gh/QuEraComputing/ParallelMergeCSR.jl)

<p>
An implementation/port of <a href="https://rd.yyrcd.com/CUDA/2022-03-14-Merge-based%20Parallel%20Sparse%20Matrix-Vector%20Multiplication.pdf"> Merrill and Garland's Merge-based Parallel Sparse Matrix-Vector Multiplication (10.1109/SC.2016.57) </a> paper in 
the &nbsp;
    <a href="https://julialang.org">
        <img src="https://raw.githubusercontent.com/JuliaLang/julia-logo-graphics/master/images/julia.ico" width="16em">
        Julia Programming Language
    </a>
    &nbsp;
</p>

ParallelMergeCSR allows you to perform *multithreaded* [CSC formatted sparse Matrix](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_.28CSC_or_CCS.29) multiplication against dense vectors and matrices as long as the sparse Matrix has had a **transpose** or **adjoint** operation applied to it via `LinearAlgebra`, built-in to Julia Base. The reason for this is the original algorithm was restricted to [CSR formatted sparse Matrices](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) but by taking the transpose of a CSC matrix you've created a CSR representation of the same matrix.

ParallelMergeCSR only exports one function: `mul!` which is used for both Sparse Matrix - Dense Vector and Sparse Matrix - Dense Matrix multiplication.

## Installation

### NOTICE
It will take *three days* before this package becomes part of the Julia registry meaning the instructions below will only work after those three days have elapsed.

To circumvent this, you can go into the Julia Package Manager and perform `add` with the URL of this repository.

## Usage

Start Julia with the desired number of threads by launching it with the `-t`/`--threads` argument:
```
julia --threads <number_of_threads>
```
or setting `JULIA_NUM_THREADS` in your environment and then running Julia.

You can confirm Julia is using the specified number of threads via:

```julia
julia> Threads.nthreads()
```

You can then use `ParallelMergeCSR` in a similar fashion to the example below:

```julia
julia> using ParallelMergeCSR, SparseArrays

# create a 20x20 transposed CSC-formatted Sparse matrix with a 30% chance of values appearing
julia> A = transpose(sprand(20, 20, 0.3));

# dense vector (can be a matrix too)
julia> B = rand(size(A, 2));

# output
julia> C = rand(size(A, 2));

# coefficients
julia> α = -1.0; β = 2.9;

# perform the operation C = ABα + Cβ, mutating C to store the answer
julia> ParallelMergeCSR.mul!(C, A, B, α, β)
```
