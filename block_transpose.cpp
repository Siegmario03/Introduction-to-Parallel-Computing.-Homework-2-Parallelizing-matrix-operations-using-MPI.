#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>

#define L2_CACHE_SIZE 1024 * 1024  // 1MB (from system info)
#define MIN_BLOCK_SIZE 8  // Ensure at least a small block for small matrices

void matTransposeBlock(std::vector<float>& matrix, int matrix_size) {
    // Compute optimal block size dynamically
    int block_size = std::sqrt(L2_CACHE_SIZE / (2 * sizeof(float)));

    // Ensure block size is within valid range
    block_size = std::min(block_size, matrix_size); // Can't exceed matrix size
    block_size = std::max(block_size, MIN_BLOCK_SIZE); // At least 8x8 blocks

    std::vector<float> transposed(matrix.size());

    // Perform blocked transposition
    for (int i = 0; i < matrix_size; i += block_size) {
        for (int j = 0; j < matrix_size; j += block_size) {
            for (int bi = 0; bi < block_size && (i + bi) < matrix_size; ++bi) {
                for (int bj = 0; bj < block_size && (j + bj) < matrix_size; ++bj) {
                    transposed[(j + bj) * matrix_size + (i + bi)] =
                        matrix[(i + bi) * matrix_size + (j + bj)];
                }
            }
        }
    }

    matrix = std::move(transposed);  // Move transposed matrix back
}




void allToAllTranspose_blocked(const std::vector<float>& matrix, int matrix_size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Compute row distribution dynamically
    int rows_per_proc = matrix_size / size;
    int extra_rows = matrix_size % size;

    // Compute send counts and displacements
    std::vector<int> send_counts(size, rows_per_proc * matrix_size);
    std::vector<int> displacements(size, 0);

    for (int i = 0; i < extra_rows; ++i) {
        send_counts[i] += matrix_size;  // Give extra row to first few processes
    }
    for (int i = 1; i < size; ++i) {
        displacements[i] = displacements[i - 1] + send_counts[i - 1];
    }

    // Allocate local matrix dynamically based on process rank
    int local_rows = send_counts[rank] / matrix_size;
    std::vector<float> local_matrix(local_rows * matrix_size);

    // Scatter using MPI_Scatterv (handles non-even row distribution)
    MPI_Scatterv(matrix.data(), send_counts.data(), displacements.data(), MPI_FLOAT,
                 local_matrix.data(), send_counts[rank], MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    // **Compute optimal block size based on L1/L2 cache**
    int block_size = std::sqrt(1024 * 1024 / (2 * sizeof(float))); // 1MB L2 cache guideline
    block_size = std::max(8, std::min(block_size, matrix_size));  // Ensure block size is within valid range

    // **Efficient Local Transposition Using Blocks**
    std::vector<float> transposed_local_matrix(local_rows * matrix_size);

    for (int i = 0; i < local_rows; i += block_size) {
        for (int j = 0; j < matrix_size; j += block_size) {
            for (int bi = 0; bi < block_size && (i + bi) < local_rows; ++bi) {
                for (int bj = 0; bj < block_size && (j + bj) < matrix_size; ++bj) {
                    transposed_local_matrix[(j + bj) * local_rows + (i + bi)] =
                        local_matrix[(i + bi) * matrix_size + (j + bj)];
                }
            }
        }
    }

    // Hierarchical Communication (All-to-All)
    int group_size = std::min(size, 4);
    int group_id = rank / group_size;
    MPI_Comm group_comm;
    MPI_Comm_split(MPI_COMM_WORLD, group_id, rank, &group_comm);

    std::vector<float> intra_group_result(local_rows * matrix_size);
    int send_count = local_rows * matrix_size / size;

    MPI_Alltoall(transposed_local_matrix.data(), send_count, MPI_FLOAT,
                 intra_group_result.data(), send_count, MPI_FLOAT, group_comm);

    // Inter-Group All-to-All Communication
    std::vector<float> final_result(matrix_size * matrix_size);
    MPI_Alltoall(intra_group_result.data(), send_count, MPI_FLOAT,
                 final_result.data(), send_count, MPI_FLOAT, MPI_COMM_WORLD);

    // Non-Blocking Communication
    MPI_Request request;
    MPI_Ialltoall(intra_group_result.data(), send_count, MPI_FLOAT,
                  transposed_local_matrix.data(), send_count, MPI_FLOAT,
                  MPI_COMM_WORLD, &request);

    // Perform dummy computation while waiting
    for (int i = 0; i < local_rows * matrix_size; ++i) {
        transposed_local_matrix[i] += 1;  // Dummy operation
    }

    MPI_Wait(&request, MPI_STATUS_IGNORE);
}


void scatterGatherTranspose_blocked(const std::vector<float>& matrix, int matrix_size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (matrix_size % size != 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rows_per_proc = matrix_size / size;
    std::vector<float> local_matrix(rows_per_proc * matrix_size);

    MPI_Barrier(MPI_COMM_WORLD);

    // Scatter rows to processes
    MPI_Scatter(matrix.data(), rows_per_proc * matrix_size, MPI_FLOAT,
                local_matrix.data(), rows_per_proc * matrix_size, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    // **Compute optimal block size based on cache size**
    int block_size = std::sqrt(1024 * 1024 / (2 * sizeof(float)));  // ~1MB L2 cache usage
    block_size = std::max(8, std::min(block_size, matrix_size)); // Ensure valid block size

    // **Efficient Local Transposition Using Blocks**
    std::vector<float> transposed_local_matrix(rows_per_proc * matrix_size);

    for (int i = 0; i < rows_per_proc; i += block_size) {
        for (int j = 0; j < matrix_size; j += block_size) {
            for (int bi = 0; bi < block_size && (i + bi) < rows_per_proc; ++bi) {
                for (int bj = 0; bj < block_size && (j + bj) < matrix_size; ++bj) {
                    transposed_local_matrix[(j + bj) * rows_per_proc + (i + bi)] =
                        local_matrix[(i + bi) * matrix_size + (j + bj)];
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Gather transposed parts back
    std::vector<float> final_result;
    if (rank == 0) final_result.resize(matrix_size * matrix_size);

    MPI_Gather(transposed_local_matrix.data(), rows_per_proc * matrix_size, MPI_FLOAT,
               final_result.data(), rows_per_proc * matrix_size, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
}

void pipelinedTranspose_blocked(const std::vector<float>& matrix, int matrix_size) {
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

    // **Compute optimal block size based on cache size**
    int block_size = std::sqrt(1024 * 1024 / (2 * sizeof(float)));  // Use ~1MB L2 cache
    block_size = std::max(8, std::min(block_size, matrix_size));  // Ensure valid size

    // **Efficient 2D Block-Based Transposition**
    for (int i = 0; i < rows_per_proc; i += block_size) {
        for (int j = 0; j < matrix_size; j += block_size) {
            for (int bi = 0; bi < block_size && (i + bi) < rows_per_proc; ++bi) {
                for (int bj = 0; bj < block_size && (j + bj) < matrix_size; ++bj) {
                    transposed_local_matrix[(j + bj) * rows_per_proc + (i + bi)] =
                        local_matrix[(i + bi) * matrix_size + (j + bj)];
                }
            }
        }
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
    std::vector<float> transposed_matrix;
    if (rank == 0) {
        transposed_matrix.resize(matrix_size * matrix_size);
    }

    MPI_Gather(transposed_local_matrix.data(), rows_per_proc * matrix_size, MPI_FLOAT,
               transposed_matrix.data(), rows_per_proc * matrix_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
}