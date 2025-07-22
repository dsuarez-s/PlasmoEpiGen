"""
Functions:
- func_toMS: Reset mosquito to susceptible state and clear parasite genomes
- mosquito_to_human: Sample parasite genome IDs for transmission from a mosquito to a human
"""

import random 
import numpy as np 
from scipy.sparse import csr_matrix  

# Reset a mosquito to susceptible and clear its parasite genomes #
def func_toMS(transition_player, X_matrix, immature_matrix, mature_matrix, MS_code):
    
    # Set the mosquito state to MS (susceptible) #
    X_matrix[transition_player] = MS_code
    
    # Remove any parasite genomes from the mature matrix #       
    mature_m = mature_matrix.tolil()
    mature_m[:, transition_player] = 0
    mature_matrix = mature_m.tocsr()

    # Remove any parasite genomes from the immature matrix #
    immature_m = immature_matrix.tolil()
    immature_m[:, transition_player] = 0
    immature_matrix = immature_m.tocsr()
    
    # Return updated state and genome matrices #
    return X_matrix, mature_matrix, immature_matrix

# ------------------------------------------------------------------------------- #
# Select parasite genomes for transmission from a mosquito vector to a human host #
# ------------------------------------------------------------------------------- #

def mosquito_to_human(X, mature_matrix, MC_code, MPC_code):
    
    # Indices of currently infected mosquitoes (monoclonal or polyclonal) #
    mosquitoes_indices = np.where((X == MC_code) | (X == MPC_code))[0]
    selected_mosquito = random.choice(mosquitoes_indices)
    
    # Extract the count vector for the selected mosquito as a dense array #
    gen_information = mature_matrix.getcol(selected_mosquito).tocoo()
    present_genomes = gen_information.row 
    weights   = gen_information.data
    
    # Raise an error if no haplotype is present (this should not happen) #
    if len(present_genomes) == 0:
        raise ValueError(f"No haplotypes found for mosquito index {selected_mosquito}")

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
    
    # Convert the result to a list #
    chosen = list(set(chosen))
    
    return chosen
    