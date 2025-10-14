"""
Functions:
- classification_S_M_PC: Classify human as Susceptible (S), Monoclonal (M), or Polyclonal (PC) based on parasite genomes
- func_lambda_humans: Apply incubation timers for exposed humans during mosquito transmission
- human_to_mosquito: Sample parasite genome IDs for transmission from a human to a mosquito
"""

import random  
import numpy as np  
from scipy.sparse import csr_matrix
import heapq


# Reset a human to susceptible and clear its parasite genomes #
def func_toHS(event_queue, transition_Player, X_matrix, immature_matrix, mature_matrix, HS_code):
    
    # Set the mosquito state to MS (susceptible) #
    X_matrix[transition_Player] = HS_code
    
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

# Select parasite genomes for transmission from a human host to a mosquito vector #
def human_to_mosquito(X, mature_matrix, HM_code, HPC_code, rng=None):
    
    rng = np.random.default_rng() if rng is None else rng

    # Indices of currently infected humans (monoclonal or polyclonal) #
    humans_indices = np.where((X == HM_code) | (X == HPC_code))[0]
    if humans_indices.size == 0:
        raise ValueError("No hay humanos infectados (HC/HPC) disponibles para transmisión.")
        
    selected_human = humans_indices[rng.integers(0, humans_indices.size)]

    col = mature_matrix.getcol(selected_human).tocoo()
    positions = col.row
    weights   = col.data
    
    if positions.size == 0:
        raise ValueError(f"No haplotypes found for human index {selected_human}")

    wsum = weights.sum()
    if wsum <= 0:
        raise ValueError(f"Pesos inválidos (suma <= 0) para el humano {selected_human}")
    probs = weights / wsum

    k = int(rng.integers(1, positions.size + 1))

    chosen = rng.choice(positions, size=k, replace=True, p=probs)

    unique_haplos = np.unique(chosen)
    return [int(i) for i in unique_haplos]