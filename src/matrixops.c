#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrixops.h" 

double dot_product(double* a, double* b, int n) {
    double result = 0.0;

    // Sequential loop with SIMD vectorization
    __m256d local_sum = _mm256_setzero_pd();

    // Process the main blocks of 4 elements
    for (int i = 0; i <= n - 4; i += 4) {
        // Load 4 doubles at once (SIMD) from both vectors a and b
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        
        // Perform element-wise multiplication
        __m256d prod = _mm256_mul_pd(va, vb);
        
        // Add the results of this block to the local sum (SIMD)
        local_sum = _mm256_add_pd(local_sum, prod);
    }

    // Reduce the local sum across the SIMD registers into a scalar result
    double sum_array[4];
    _mm256_storeu_pd(sum_array, local_sum);
    result += sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

    // Handle any remaining elements that don't fit into a full 4-element block
    for (int i = n - n % 4; i < n; ++i) {
        result += a[i] * b[i];
    }

    return result;
}