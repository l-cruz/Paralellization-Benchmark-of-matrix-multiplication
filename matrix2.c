#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define RUNS 5

double **generateDense(int N) {
    double **M = malloc(N * sizeof(double*));
    for (int i = 0; i < N; i++) {
        M[i] = malloc(N * sizeof(double));
        for (int j = 0; j < N; j++)
            M[i][j] = (double)rand() / RAND_MAX;
    }
    return M;
}

double **generateSparse(int N, double sparsity) {
    double density = 1 - sparsity;
    double **M = malloc(N * sizeof(double*));

    for (int i = 0; i < N; i++) {
        M[i] = malloc(N * sizeof(double));
        for (int j = 0; j < N; j++) {
            double r = (double)rand() / RAND_MAX;
            if (r < density)
                M[i][j] = (double)rand() / RAND_MAX;
            else
                M[i][j] = 0.0;
        }
    }
    return M;
}

void freeMatrix(double **M, int N) {
    for (int i = 0; i < N; i++)
        free(M[i]);
    free(M);
}

void multiplyDense(double **A, double **B, double **C, int N) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < N; k++)
                sum += A[i][k] * B[k][j];

            C[i][j] = sum;
        }
}

void multiplySparse(double **A, double **B, double **C, int N) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {

            double sum = 0;
            for (int k = 0; k < N; k++)
                if (A[i][k] != 0 && B[k][j] != 0)
                    sum += A[i][k] * B[k][j];

            C[i][j] = sum;
        }
}

void multiplyOptimized(double **A, double **B, double **C, int N) {

    double **Bt = malloc(N * sizeof(double*));

    for (int i = 0; i < N; i++) {
        Bt[i] = malloc(N * sizeof(double));
        for (int j = 0; j < N; j++)
            Bt[i][j] = B[j][i];
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {

            double sum = 0;
            for (int k = 0; k < N; k++)
                sum += A[i][k] * Bt[j][k];

            C[i][j] = sum;
        }

    for (int i = 0; i < N; i++)
        free(Bt[i]);
    free(Bt);
}

int counter = 0;

void demoSync() {
    #pragma omp parallel for
    for (int i = 0; i < 10000; i++) {
        #pragma omp critical
        counter++;
    }
}

int main() {

    srand(time(NULL));

    int sizes[] = {64,128,256,512,768,1024};
    double sparsities[] = {0.1, 0.5, 0.9};

    system("mkdir results >nul 2>&1");

    FILE *f = fopen("results/c_parallel.txt", "w");

    for (int s = 0; s < 6; s++) {

        int N = sizes[s];
        fprintf(f, "\n=== Dense %dx%d ===\n", N, N);

        double sumTotal = 0;

        for (int r = 1; r <= RUNS; r++) {

            double **A = generateDense(N);
            double **B = generateDense(N);
            double **C = generateDense(N);

            double start = omp_get_wtime();
            multiplyDense(A, B, C, N);
            double end = omp_get_wtime();

            double elapsed = end - start;
            sumTotal += elapsed;

            fprintf(f, "Run %d: %.3f s\n", r, elapsed);

            freeMatrix(A, N);
            freeMatrix(B, N);
            freeMatrix(C, N);
        }

        fprintf(f, "Mean: %.3f s\n", sumTotal / RUNS);
    }

    for (int sp = 0; sp < 3; sp++) {

        double spars = sparsities[sp];
        fprintf(f, "\n=== Sparse (sparsity = %.1f) ===\n", spars);

        for (int s = 0; s < 6; s++) {

            int N = sizes[s];
            fprintf(f, "--- N=%d ---\n", N);

            double sumTotal = 0;

            for (int r = 1; r <= RUNS; r++) {

                double **A = generateSparse(N, spars);
                double **B = generateSparse(N, spars);
                double **C = generateDense(N);

                double start = omp_get_wtime();
                multiplySparse(A, B, C, N);
                double end = omp_get_wtime();

                double elapsed = end - start;
                sumTotal += elapsed;

                fprintf(f, "Run %d: %.3f s\n", r, elapsed);

                freeMatrix(A, N);
                freeMatrix(B, N);
                freeMatrix(C, N);
            }

            fprintf(f, "Mean: %.3f s\n", sumTotal / RUNS);
        }
    }

    for (int s = 0; s < 6; s++) {

        int N = sizes[s];
        fprintf(f, "\n=== Optimized %dx%d ===\n", N, N);

        double sumTotal = 0;

        for (int r = 1; r <= RUNS; r++) {

            double **A = generateDense(N);
            double **B = generateDense(N);
            double **C = generateDense(N);

            double start = omp_get_wtime();
            multiplyOptimized(A, B, C, N);
            double end = omp_get_wtime();

            double elapsed = end - start;
            sumTotal += elapsed;

            fprintf(f, "Run %d: %.3f s\n", r, elapsed);

            freeMatrix(A, N);
            freeMatrix(B, N);
            freeMatrix(C, N);
        }

        fprintf(f, "Mean: %.3f s\n", sumTotal / RUNS);
    }

    fclose(f);
    demoSync();
    printf("Synchronized counter: %d\n", counter);

    return 0;
}
