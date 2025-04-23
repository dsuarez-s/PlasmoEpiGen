"""
Functions:
- func_toMS: Reset mosquito to susceptible state and clear parasite genomes
- mosquito_to_human: Sample parasite genome IDs for transmission from a mosquito to a human
"""

import random 
import numpy as np 
from scipy.sparse import csr_matrix  

# States: HS=0, HM=1, HPC=2, MS=3, MC=4, MPC=5 #
HS, HM, HPC = 0, 1, 2
MS, MC, MPC = 3, 4, 5

# Reset a mosquito to susceptible and clear its parasite genomes #
def func_toMS(transition_player: int, X_matrix: np.ndarray,
              pre_genomes_matrix: csr_matrix, 
              genomes_matrix: csr_matrix) -> tuple[np.ndarray, csr_matrix, csr_matrix]:
    
    # Set the mosquito state to MS (susceptible) #
    X_matrix[transition_player] = MS

    # Remove any parasite genomes from the main genomes matrix #
    if genomes_matrix.getnnz() > 0:
        genomes_matrix[:, transition_player] = 0
        genomes_matrix.eliminate_zeros()

    # Remove any parasite genomes from the pre-infection matrix #
    if pre_genomes_matrix.getnnz() > 0:
        pre_genomes_matrix[:, transition_player] = 0
        pre_genomes_matrix.eliminate_zeros()

    # Return updated state and genome matrices #
    return X_matrix, genomes_matrix, pre_genomes_matrix

# Select parasite genomes for transmission from a mosquito vector to a human host #
def mosquito_to_human( X: np.ndarray, genomes_matrix: csr_matrix,
                      MC: int, MPC: int) -> list[int]:
    
    # indices of currently infected mosquitoes (monoclonal or polyclonal) #
    mosquito_indices = np.where((X == MC) | (X == MPC))[0]
    selected_mosquito = random.choice(mosquito_indices)

    # present_genomes: genome IDs present in the selected mosquito #
    present_genomes = genomes_matrix[:, selected_mosquito].tocoo().row
    if len(present_genomes) <= 1:
        return list(present_genomes)

    # Sample a random subset of genomes for transmission #
    return random.sample(list(present_genomes), random.randint(1, len(present_genomes)))
