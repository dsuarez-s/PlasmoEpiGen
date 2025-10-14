"""
Functions:
- func_toMS: Reset mosquito to susceptible state and clear parasite genomes
- mosquito_to_human: Sample parasite genome IDs for transmission from a mosquito to a human
"""

import random 
import numpy as np 
from scipy.sparse import csr_matrix  
import heapq

# Reset a mosquito to susceptible and clear its parasite genomes #
def func_toMS(event_queue, transition_Player, X_matrix, immature_matrix, mature_matrix, MS_code):
    
    # Set the mosquito state to MS (susceptible) #
    X_matrix[transition_Player] = MS_code
    
    # Remove any parasite genomes from the mature matrix #       
    mature_m = mature_matrix.tolil()
    mature_m[:, transition_Player] = 0
    mature_matrix = mature_m.tocsr()

    # Remove any parasite genomes from the immature matrix #
    immature_m = immature_matrix.tolil()
    immature_m[:, transition_Player] = 0
    immature_matrix = immature_m.tocsr()
    
    event_queue = [evt for evt in event_queue if evt[3] != transition_Player]
    heapq.heapify(event_queue)
    
    # Return updated state and genome matrices #
    return X_matrix, mature_matrix, immature_matrix, event_queue

# ------------------------------------------------------------------------------- #
# Select parasite genomes for transmission from a mosquito vector to a human host #
# ------------------------------------------------------------------------------- #

def mosquito_to_human(X, mature_matrix, MC_code, MPC_code, rng=None):
    
    rng = np.random.default_rng() if rng is None else rng
    
    # Indices of currently infected mosquitoes (monoclonal or polyclonal) #
    mosquitoes_indices = np.where((X == MC_code) | (X == MPC_code))[0]
    
    if mosquitoes_indices.size == 0:
        raise ValueError("No hay mosquitos infectados (MC/MPC) disponibles para transmisión.")
        
    selected_mosquito = random.choice(mosquitoes_indices)
    
    # Columna (conteos de haplotipos) del mosquito elegido
    col = mature_matrix.getcol(selected_mosquito).tocoo()
    positions = col.row        # np.array de índices de haplotipos presentes
    weights   = col.data       # np.array de pesos/conteos
    
    # Raise an error if no haplotype is present (this should not happen) #
    if positions.size == 0:
        raise ValueError(f"No haplotypes found for mosquito index {selected_mosquito}")
        
    wsum = weights.sum()
    if wsum <= 0:
        raise ValueError(f"Pesos inválidos (suma <= 0) para el mosquito {selected_mosquito}")
    probs = weights / wsum
    
    # k aleatorio en [1, len(positions)]
    k = int(rng.integers(1, positions.size + 1))
    
    # Si solo hay 1 haplotipo, muestreo con reemplazo y luego únicos -> 1 elemento
    chosen = rng.choice(positions, size=k, replace=True, p=probs)
    
    # Reportar únicos (ordenados). Si prefieres “orden estable”, te paso abajo una variante.
    unique_haplos = np.unique(chosen)
    return [int(i) for i in unique_haplos]
    