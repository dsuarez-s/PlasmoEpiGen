# Mosquito state update functions #
import numpy as np

# States: HS=0, HM=1, HPC=2, MS=3, MC=4, MPC=5 #
HS, HM, HPC = 0, 1, 2
MS, MC, MPC = 3, 4, 5

def func_toMS(transitionPlayer, X_matrix,
              pre_genomes_matrix, genomes_matrix):
    # Reset a mosquito to susceptible and clear its parasites #
    X_matrix[transitionPlayer] = MS

    if genomes_matrix.getnnz() > 0:
        genomes_matrix[:, transitionPlayer] = 0
        genomes_matrix.eliminate_zeros()

    if pre_genomes_matrix.getnnz() > 0:
        pre_genomes_matrix[:, transitionPlayer] = 0
        pre_genomes_matrix.eliminate_zeros()

    return [X_matrix, genomes_matrix, pre_genomes_matrix]
