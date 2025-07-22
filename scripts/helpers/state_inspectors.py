import numpy as np
from scipy.sparse import csr_matrix

def classification_S_M_PC(transitionPlayer, X_matrix, mature_matrix,
                          HS_code = 0, HM_code = 1, HPC_code = 2, 
                          MS_code = 3, MC_code = 4, MPC_code = 5):
    
    humans_states = {HS_code, HM_code, HPC_code}
    state = X_matrix[transitionPlayer]
    # Update agent state based on genome presence #
    num_genomes = mature_matrix.getcol(transitionPlayer).count_nonzero()

    if num_genomes == 0:
        X_matrix[transitionPlayer] = HS_code if state in humans_states else MS_code
    elif num_genomes == 1:
        X_matrix[transitionPlayer] = HM_code if state in humans_states else MC_code
    else:
        X_matrix[transitionPlayer] = HPC_code if state in humans_states else MPC_code
        
    return(X_matrix)
        
# -------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------- #

def update_matrices(mature_matrix, immature_matrix, parasitic_populations):
    # Remove extinct haplotypes from matrices #
    combined = mature_matrix + immature_matrix
    alive = np.asarray(combined.sum(axis=1)).squeeze() != 0
    return (parasitic_populations[alive], mature_matrix[alive], immature_matrix[alive])

# -------------------------------------------------------------------------------------- #