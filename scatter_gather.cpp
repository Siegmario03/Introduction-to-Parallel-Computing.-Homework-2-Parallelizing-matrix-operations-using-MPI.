#include <mpi.h>
#include <vector>
#include <iostream>
#include <cmath>
#include "scatter_gather.h"

void scatterGatherTranspose(const std::vector<float>& matrix, int matrix_size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (matrix_size % size != 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rows_per_proc = matrix_size / size;
    std::vector<float> local_matrix(rows_per_proc * matrix_size);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Scatter(matrix.data(), rows_per_proc * matrix_size, MPI_FLOAT,
                local_matrix.data(), rows_per_proc * matrix_size, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<float> transposed_local_matrix(rows_per_proc * matrix_size);
    for (int i = 0; i < rows_per_proc; ++i)
        for (int j = 0; j < matrix_size; ++j)
            transposed_local_matrix[j * rows_per_proc + i] = local_matrix[i * matrix_size + j];

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<float> final_result;
    if (rank == 0) final_result.resize(matrix_size * matrix_size);

    MPI_Gather(transposed_local_matrix.data(), rows_per_proc * matrix_size, MPI_FLOAT,
               final_result.data(), rows_per_proc * matrix_size, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    
}