#include "matrixops.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    int n = 1000;
    double *a = aligned_alloc(32, n * sizeof(double));
    double *b = aligned_alloc(32, n * sizeof(double));

    for (int i = 0; i < n; ++i) {
        a[i] = 1.0;
        b[i] = 2.0;
    }
    
    double start = omp_get_wtime();
    double result = dot_product(a, b, n);
    double end = omp_get_wtime();

    printf("Dot Product: %.2f\n", result);
    printf("Time Taken (C with AVX: ): %f seconds\n", end - start);

    free(a);
    free(b);

    return 0;
}

// -----------------------------------------------------

// for benchmark
// int main(int argc, char *argv[]) {
//     if (argc < 2) {
//         printf("Usage: %s <size_of_vector>\n", argv[0]);
//         return 1; // Exit with error if the argument is not provided
//     }

//     // Read vector size n from command line argument
//     int n = atoi(argv[1]);
//     if (n <= 0) {
//         printf("Invalid size provided. Size must be a positive integer.\n");
//         return 1;
//     }

//     // Allocate aligned memory for vectors a and b
//     double *a = aligned_alloc(32, n * sizeof(double));
//     double *b = aligned_alloc(32, n * sizeof(double));

//     // Initialize vectors a and b with example values
//     for (int i = 0; i < n; ++i) {
//         a[i] = 1.0;
//         b[i] = 2.0;
//     }

//     // Measure the start time
//     double start = omp_get_wtime();

//     // Compute the dot product
//     double result = dot_product(a, b, n);

//     // Measure the end time
//     double end = omp_get_wtime();

//     // Print the result and the time taken
//     printf("Dot Product: %.2f\n", result);
//     printf("Time Taken (C with AVX + OMP): %f seconds\n", end - start);

//     // Free the allocated memory
//     free(a);
//     free(b);

//     return 0;
// }