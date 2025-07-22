"""
Functions:
- classification_S_M_PC: Classify human as Susceptible (S), Monoclonal (M), or Polyclonal (PC) based on parasite genomes
- func_lambda_humans: Apply incubation timers for exposed humans during mosquito transmission
- human_to_mosquito: Sample parasite genome IDs for transmission from a human to a mosquito
"""

import random  
import numpy as np  
from scipy.sparse import csr_matrix


# Select parasite genomes for transmission from a human host to a mosquito vector #
def human_to_mosquito(X, mature_matrix, HM_code, HPC_code):
    
    # Indices of currently infected humans (monoclonal or polyclonal) #
    humans_indices = np.where((X == HM_code) | (X == HPC_code))[0]
    selected_human = random.choice(humans_indices)
    
    # Extract the count vector for the selected mosquito as a dense array #
    gen_information = mature_matrix.getcol(selected_human).tocoo()
    present_genomes = gen_information.row 
    weights   = gen_information.data
    
    # Raise an error if no haplotype is present (this should not happen) #
    if len(present_genomes) == 0:
        raise ValueError(f"No haplotypes found for human index {selected_human}")

    # If exactly one haplotype is present, return it directly #
    if len(present_genomes) == 1:
        return [int(present_genomes[0])]
    
    # Normalize weights to sum to 1 to obtain probabilities #
    probabilities = weights / weights.sum()

    # Convert indices and probabilities to lists #
    positions = list(present_genomes)
    probs = list(probabilities)

    # Randomly choose how many haplotypes to transmit #
    k = np.random.randint(1, len(positions) + 1)

    # Sample k unique haplotypes weighted by their probabilities #
    chosen = np.random.choice(positions, size=k, replace=True, p=probs)
    
    # Convert the result to a list and delete duplicates #
    chosen = list(set(chosen))
    
    return chosen