#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrixops.h" 

// Dot product with AVX2, etc..
double dot_product(double *a, double *b, int n) {
    double result = 0.0;
    
    // Use signed integer for OpenMP loop Just to make sure that works :)
    int i = 0;
    
    // Use OpenMP for multithreading
    #pragma omp parallel for reduction(+:result)
    for (i = 0; i < n - 7; i += 8) {
        // Load 8 doubles from each vector (AVX2)
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        
        // Perform element-wise multiplication
        __m256d vprod = _mm256_mul_pd(va, vb);
        
        // For doubles, we need a different approach for horizontal addition
        // First, compute partial sums in the high and low lanes
        __m256d sum = _mm256_hadd_pd(vprod, vprod);
        
        // Extract the two doubles from the AVX register
        double sum_array[4];
        _mm256_storeu_pd(sum_array, sum);
        result += sum_array[0] + sum_array[2];
        
        // Process the next 4 elements
        va = _mm256_loadu_pd(&a[i+4]);
        vb = _mm256_loadu_pd(&b[i+4]);
        vprod = _mm256_mul_pd(va, vb);
        sum = _mm256_hadd_pd(vprod, vprod);
        _mm256_storeu_pd(sum_array, sum);
        result += sum_array[0] + sum_array[2];
    }
    
    // Handle remaining elements
    for (int j = i; j < n; ++j) {
        result += a[j] * b[j];
    }
    
    return result;
}