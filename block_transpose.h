#ifndef BLOCK_TRANSPOSE_H
#define BLOCK_TRANSPOSE_H

void matTransposeBlock(std::vector<float>& matrix, int matrix_size);
void allToAllTranspose_blocked(const std::vector<float>& matrix, int matrix_size);
void scatterGatherTranspose_blocked(const std::vector<float>& matrix, int matrix_size);
void pipelinedTranspose_blocked(const std::vector<float>& matrix, int matrix_size);

#endif // BLOCK_TRANSPOSE_H