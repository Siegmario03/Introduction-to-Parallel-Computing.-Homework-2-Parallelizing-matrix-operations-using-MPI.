#include <vector>
#include <iostream>
#include "sequential.h"

void Sequential_transpose(const std::vector<float>& matrix, int matrix_size) {
    std::vector<float> transposed_matrix(matrix_size * matrix_size);

    for (int i = 0; i < matrix_size; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            transposed_matrix[j * matrix_size + i] = matrix[i * matrix_size + j];
        }
    }
}