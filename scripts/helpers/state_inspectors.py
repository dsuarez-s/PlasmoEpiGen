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

# def update_matrices(mature_matrix, immature_matrix, parasitic_populations):
    
#     # Determinar filas vivas #
#     alive_m = np.asarray(mature_matrix.getnnz(axis=1)).ravel()
#     alive_i = np.asarray(immature_matrix.getnnz(axis=1)).ravel()
#     alive = (alive_m + alive_i) > 0

#     n_rows = alive.shape[0]

#     # Mapa de índices: -1 para filas podadas; 0..n_alive-1 para filas vivas
#     old_to_new = np.full(n_rows, -1, dtype=np.int64)
#     old_to_new[alive] = np.arange(int(alive.sum()), dtype=np.int64)

#     return (parasitic_populations[alive], mature_matrix[alive], immature_matrix[alive], old_to_new )

# -------------------------------------------------------------------------------------- #