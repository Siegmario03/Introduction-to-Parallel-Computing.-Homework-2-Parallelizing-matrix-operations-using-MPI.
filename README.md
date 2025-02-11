# Introduction-to-Parallel-Computing.-Homework-2-Parallelizing-matrix-operations-using-MPI.

# MPI Transpose Project

This project demonstrates various methods for matrix transposition using MPI (Message Passing Interface). The methods include Sequential, All-to-All, Scatter/Gather, Pipelined, and their blocked versions.

## Prerequisites

- GCC 9.1.0
- MPICH 3.2.1

## Setup

1. Clone the repository to your local machine:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Load the required modules:
    ```bash
    module load gcc/9.1.0
    module load mpich/3.2.1--gcc-9.1.0
    ```

## Compilation

Compile the source code using the following command:
```bash
mpic++ -o main main.cpp all_toall.cpp scatter_gather.cpp pipelined.cpp sequential.cpp block_transpose.cpp
```

## Execution
To run the project, use the provided PBS script. Submit the job to the cluster using the following command:
```bash
qsub run_mpi_transpose.pbs
```
The PBS script run_mpi_transpose.pbs will compile the code and execute the program with different numbers of processes (1, 2, 4, 8, 16).
