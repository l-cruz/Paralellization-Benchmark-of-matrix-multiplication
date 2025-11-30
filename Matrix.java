import java.io.FileWriter;
import java.io.IOException;
import java.util.stream.IntStream;

public class Matrix {

    static int RUNS = 5;
    static double[][] randomDense(int N) {
        double[][] M = new double[N][N];
        for(int i=0;i<N;i++)
            for(int j=0;j<N;j++)
                M[i][j] = Math.random();
        return M;
    }

    static double[][] randomSparse(int N, double sparsity) {
        double[][] M = new double[N][N];
        double density = 1.0 - sparsity;

        for(int i=0;i<N;i++)
            for(int j=0;j<N;j++)
                M[i][j] = (Math.random() < density) ? Math.random() : 0.0;

        return M;
    }

    static double[][] multiplyDenseParallel(double[][] A, double[][] B) {
        int N = A.length;
        double[][] C = new double[N][N];

        IntStream.range(0, N).parallel().forEach(i -> {
            for (int j = 0; j < N; j++) {
                double sum = 0;
                for (int k = 0; k < N; k++)
                    sum += A[i][k] * B[k][j];
                C[i][j] = sum;
            }
        });
        return C;
    }

    static double[][] multiplySparse(double[][] A, double[][] B) {
        int N = A.length;
        double[][] C = new double[N][N];

        for(int i=0;i<N;i++) {
            for(int j=0;j<N;j++) {
                double sum = 0;
                for(int k=0;k<N;k++) {
                    if(A[i][k] != 0 && B[k][j] != 0)
                        sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
        return C;
    }

    static double[][] multiplyOptimized(double[][] A, double[][] B) {
        int N = A.length;
        double[][] Bt = new double[N][N];

        // transpose B
        for(int i=0;i<N;i++)
            for(int j=0;j<N;j++)
                Bt[i][j] = B[j][i];

        double[][] C = new double[N][N];

        for(int i=0;i<N;i++)
            for(int j=0;j<N;j++){
                double sum=0;
                for(int k=0;k<N;k++)
                    sum += A[i][k] * Bt[j][k];
                C[i][j]=sum;
            }

        return C;
    }

    public static void main(String[] args) throws IOException {

        int[] sizes = {64,128,256,512,768,1024};
        double[] sparsityLevels = {0.1,0.5,0.9};

        new java.io.File("results").mkdirs();
        FileWriter fw = new FileWriter("results/java_parallel.txt");

        for(int N : sizes) {
            fw.write("\n=== Dense " + N + "x" + N + " ===\n");

            double total=0;
            for(int r=1;r<=RUNS;r++){
                double[][] A = randomDense(N);
                double[][] B = randomDense(N);

                long start = System.nanoTime();
                multiplyDenseParallel(A,B);
                long end = System.nanoTime();

                double sec = (end-start)/1e9;
                total += sec;

                fw.write("Run " + r + ": " + sec + " s\n");
            }
            fw.write("Mean: " + (total/RUNS) + " s\n");
        }

        for(double sp : sparsityLevels) {

            fw.write("\n=== Sparse (sparsity = " + sp + ") ===\n");

            for(int N : sizes) {

                fw.write("--- N=" + N + " ---\n");

                double total=0;

                for(int r=1;r<=RUNS;r++){
                    double[][] A = randomSparse(N, sp);
                    double[][] B = randomSparse(N, sp);

                    long start = System.nanoTime();
                    multiplySparse(A,B);
                    long end = System.nanoTime();

                    double sec = (end-start)/1e9;
                    total += sec;

                    fw.write("Run " + r + ": " + sec + " s\n");
                }

                fw.write("Mean: " + (total/RUNS) + " s\n");
            }
        }

        for(int N : sizes) {

            fw.write("\n=== Optimized " + N + "x" + N + " ===\n");

            double total=0;

            for(int r=1;r<=RUNS;r++){
                double[][] A = randomDense(N);
                double[][] B = randomDense(N);

                long start = System.nanoTime();
                multiplyOptimized(A,B);
                long end = System.nanoTime();

                double sec = (end-start)/1e9;
                total += sec;

                fw.write("Run " + r + ": " + sec + " s\n");
            }

            fw.write("Mean: " + (total/RUNS) + " s\n");
        }

        fw.close();
        System.out.println("Java results generated in results/java_parallel.txt");
    }
}