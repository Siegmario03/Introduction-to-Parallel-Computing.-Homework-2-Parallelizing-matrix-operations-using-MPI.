#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <functional>
#include <cstdlib>
#include "all_toall.h"
#include "scatter_gather.h"
#include "pipelined.h"
#include "sequential.h"
#include "block_transpose.h"

const int NUM_RUNS = 30;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Define matrix sizes to test
    std::vector<int> matrix_sizes = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

    for (int matrix_size : matrix_sizes) {
        std::vector<float> matrix;

        if (rank == 0) {
            // Initialize matrix with random values
            matrix.resize(matrix_size * matrix_size);
            std::srand(std::time(0));
            for (int i = 0; i < matrix_size; ++i)
                for (int j = 0; j < matrix_size; ++j)
                    matrix[i * matrix_size + j] = static_cast<float>(std::rand()) / RAND_MAX;
        }

        // Broadcast matrix size and data to all processes
        MPI_Bcast(&matrix_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            matrix.resize(matrix_size * matrix_size);
        }
        MPI_Bcast(matrix.data(), matrix_size * matrix_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

        // Compute function execution time
        auto computeAverage = [&](std::function<void()> func) {
            double total_time = 0.0;
            for (int i = 0; i < NUM_RUNS; ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                func();
                auto end = std::chrono::high_resolution_clock::now();
                total_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            }
            return total_time / NUM_RUNS;
        };

        // Measure time for each method
        double avg_sequential = computeAverage([&]() { Sequential_transpose(matrix, matrix_size); });
        double avg_alltoall = computeAverage([&]() { allToAllTranspose(matrix, matrix_size); });
        double avg_alltoall_blocked = computeAverage([&]() { allToAllTranspose_blocked(matrix, matrix_size); });
        double avg_scatter = computeAverage([&]() { scatterGatherTranspose(matrix, matrix_size); });
        double avg_scatter_blocked = computeAverage([&]() { scatterGatherTranspose_blocked(matrix, matrix_size); });
        double avg_pipelined = computeAverage([&]() { pipelinedTranspose(matrix, matrix_size); });
        double avg_pipelined_blocked = computeAverage([&]() { pipelinedTranspose_blocked(matrix, matrix_size); });

        //bool symmetric = isSymmetric(matrix, matrix_size);

        if (rank == 0) {
            // Display performance comparison
            std::cout << "Matrix Size: " << matrix_size << " x " << matrix_size << std::endl;
            std::cout << "---------------------------------------------------" << std::endl;
            std::cout << "Sequential Transpose Time          : " << avg_sequential << " ms" << std::endl;
            std::cout << "All-to-All Transpose Time          : " << avg_alltoall << " ms" << std::endl;
            std::cout << "All-to-All (Blocked) Time          : " << avg_alltoall_blocked << " ms" << std::endl;
            std::cout << "Scatter-Gather Transpose Time      : " << avg_scatter << " ms" << std::endl;
            std::cout << "Scatter-Gather (Blocked) Time      : " << avg_scatter_blocked << " ms" << std::endl;
            std::cout << "Pipelined Transpose Time           : " << avg_pipelined << " ms" << std::endl;
            std::cout << "Pipelined (Blocked) Transpose Time : " << avg_pipelined_blocked << " ms" << std::endl;
            //std::cout << "---------------------------------------------------" << std::endl;
            //std::cout << "Matrix Symmetry Check              : " << (symmetric ? "YES" : "NO") << std::endl;
            std::cout << "===================================================" << std::endl;
        }

        // Ensure all processes finish before next iteration
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
