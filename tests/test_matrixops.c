#include "matrixops.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    int n = 1000000;  // Number of elements in the vectors
    double *a = (double *)malloc(n * sizeof(double));
    double *b = (double *)malloc(n * sizeof(double));

    // Initialize the vectors with random values
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        a[i] = (double)rand() / RAND_MAX;
        b[i] = (double)rand() / RAND_MAX;
    }

    // Measure the time taken for the dot product computation
    clock_t start_time = clock();
    double result = dot_product(a, b, n);
    clock_t end_time = clock();

    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Dot product result: %f\n", result);
    printf("Elapsed time: %f seconds\n", elapsed_time);

    free(a);
    free(b);

    return 0;
}