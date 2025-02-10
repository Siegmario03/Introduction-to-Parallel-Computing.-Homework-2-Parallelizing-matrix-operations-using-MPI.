#include <mpi.h>
#include <vector>
#include <iostream>
#include "pipelined.h"

void pipelinedTranspose(const std::vector<float>& matrix, int matrix_size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Determine rows assigned to each process
    int rows_per_proc = matrix_size / size;
    int remainder = matrix_size % size;
    if (rank < remainder) rows_per_proc++;  // Distribute extra rows

    // Allocate memory for local matrix
    std::vector<float> local_matrix(rows_per_proc * matrix_size, 0);
    std::vector<float> transposed_local_matrix(rows_per_proc * matrix_size, 0);

    // Scatter the matrix to all processes
    MPI_Scatter(matrix.data(), rows_per_proc * matrix_size, MPI_FLOAT,
                local_matrix.data(), rows_per_proc * matrix_size, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // Transpose local matrix efficiently
    for (int i = 0; i < rows_per_proc; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            transposed_local_matrix[j * rows_per_proc + i] = local_matrix[i * matrix_size + j];
        }
    }

    // Allocate global transposed matrix (only in root)
    std::vector<float> transposed_matrix;
    if (rank == 0) {
        transposed_matrix.resize(matrix_size * matrix_size);
    }

    // **Optimized Pipelined Communication**
    int send_to = (rank + 1) % size;
    int recv_from = (rank - 1 + size) % size;

    std::vector<float> recv_buffer(rows_per_proc * matrix_size);
    MPI_Request send_req, recv_req;

    for (int step = 0; step < size - 1; ++step) {
        // Non-blocking send and receive
        MPI_Isend(transposed_local_matrix.data(), rows_per_proc * matrix_size, MPI_FLOAT, send_to, 0, MPI_COMM_WORLD, &send_req);
        MPI_Irecv(recv_buffer.data(), rows_per_proc * matrix_size, MPI_FLOAT, recv_from, 0, MPI_COMM_WORLD, &recv_req);

        // Wait for communication to complete
        MPI_Wait(&send_req, MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

        // Swap buffers for next step
        transposed_local_matrix.swap(recv_buffer);
    }

    // Gather transposed results at root
    MPI_Gather(transposed_local_matrix.data(), rows_per_proc * matrix_size, MPI_FLOAT,
               transposed_matrix.data(), rows_per_proc * matrix_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

}