#include <mpi.h>
#include <vector>
#include <iostream>
#include "all_toall.h"

void allToAllTranspose(const std::vector<float>& matrix, int matrix_size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ? Compute row distribution dynamically
    int rows_per_proc = matrix_size / size;
    int extra_rows = matrix_size % size;
    
    // ? Compute send counts and displacements
    std::vector<int> send_counts(size, rows_per_proc * matrix_size);
    std::vector<int> displacements(size, 0);

    for (int i = 0; i < extra_rows; ++i) {
        send_counts[i] += matrix_size;  // Give extra row to first few processes
    }
    for (int i = 1; i < size; ++i) {
        displacements[i] = displacements[i - 1] + send_counts[i - 1];
    }

    // ? Allocate local matrix dynamically based on process rank
    int local_rows = send_counts[rank] / matrix_size;
    std::vector<float> local_matrix(local_rows * matrix_size);

    // ? Scatter using MPI_Scatterv (handles non-even row distribution)
    MPI_Scatterv(matrix.data(), send_counts.data(), displacements.data(), MPI_FLOAT,
                 local_matrix.data(), send_counts[rank], MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    // ? Step 2: Efficient Local Transposition
    std::vector<float> transposed_local_matrix(local_rows * matrix_size);
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            transposed_local_matrix[j * local_rows + i] = local_matrix[i * matrix_size + j];
        }
    }

    // ? Step 3: Hierarchical Communication
    int group_size = std::min(size, 4);  // Ensures groups are well-formed
    int group_id = rank / group_size;
    MPI_Comm group_comm;
    MPI_Comm_split(MPI_COMM_WORLD, group_id, rank, &group_comm);

    std::vector<float> intra_group_result(local_rows * matrix_size);
    int send_count = local_rows * matrix_size / size;  // Fixed size issue!

    // ? **All processes must call MPI_Alltoall** to prevent deadlock
    MPI_Alltoall(transposed_local_matrix.data(), send_count, MPI_FLOAT,
                 intra_group_result.data(), send_count, MPI_FLOAT, group_comm);

    // ? Step 4: Inter-Group All-to-All (Ensure all groups participate!)
    std::vector<float> final_result(matrix_size * matrix_size);
    MPI_Alltoall(intra_group_result.data(), send_count, MPI_FLOAT,
                 final_result.data(), send_count, MPI_FLOAT, MPI_COMM_WORLD);

    // ? Step 5: Non-Blocking Communication
    MPI_Request request;
    MPI_Ialltoall(intra_group_result.data(), send_count, MPI_FLOAT,
                  transposed_local_matrix.data(), send_count, MPI_FLOAT,
                  MPI_COMM_WORLD, &request);

    // ? Perform useful computation while waiting
    for (int i = 0; i < local_rows * matrix_size; ++i) {
        transposed_local_matrix[i] += 1;  // Dummy operation
    }

    MPI_Wait(&request, MPI_STATUS_IGNORE);
}

bool isSymmetric(const std::vector<float>& matrix, int matrix_size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ? Compute row distribution dynamically
    int rows_per_proc = matrix_size / size;
    int extra_rows = matrix_size % size;

    // ? Compute send counts and displacements
    std::vector<int> send_counts(size, rows_per_proc * matrix_size);
    std::vector<int> displacements(size, 0);

    for (int i = 0; i < extra_rows; ++i) {
        send_counts[i] += matrix_size;
    }
    for (int i = 1; i < size; ++i) {
        displacements[i] = displacements[i - 1] + send_counts[i - 1];
    }

    // ? Allocate local matrix dynamically
    int local_rows = send_counts[rank] / matrix_size;
    std::vector<float> local_matrix(local_rows * matrix_size);

    // ? Scatter the matrix rows to all processes
    MPI_Scatterv(matrix.data(), send_counts.data(), displacements.data(), MPI_FLOAT,
                 local_matrix.data(), send_counts[rank], MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    // ? Step 2: Check symmetry locally
    bool local_symmetric = true;
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            if (local_matrix[i * matrix_size + j] != matrix[j * matrix_size + (displacements[rank] / matrix_size) + i]) {
                local_symmetric = false;
                break;
            }
        }
        if (!local_symmetric) break;
    }

    int global_symmetric;
    int local_result = local_symmetric ? 1 : 0;
    MPI_Allreduce(&local_result, &global_symmetric, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    return global_symmetric == 1;
}
