"""
Functions:
- classification_S_M_PC: Classify human as Susceptible (S), Monoclonal (M), or Polyclonal (PC) based on parasite genomes
- func_lambda_humans: Apply incubation timers for exposed humans during mosquito transmission
- human_to_mosquito: Sample parasite genome IDs for transmission from a human to a mosquito
"""

import random  
import numpy as np  
from scipy.sparse import csr_matrix

# Determine infection classification of a human agent based on its parasite genomes #
def classification_S_M_PC(transition_player: int, genomes_matrix: csr_matrix) -> str:
    # player_genomes: genome IDs present in the specified human agent #
    player_genomes = genomes_matrix[:, transition_player].tocoo().row
    if len(player_genomes) == 0:
        return "S"
    elif len(player_genomes) == 1:
        return "M"
    else:
        return "PC"

# Set incubation timers for exposed humans when infected by mosquitoes #
def func_lambda_humans(transition_player: int,
                       inoculated_genomes: list,
                       pre_genomes_matrix: csr_matrix,
                       X_matrix: np.ndarray,
                       gamma: float) -> None:
    # For each inoculated genome ID, set the timer until it becomes infectious #
    for genome_id in inoculated_genomes:
        pre_genomes_matrix[genome_id, transition_player] = gamma

# Select parasite genomes for transmission from a human host to a mosquito vector #
def human_to_mosquito(X: np.ndarray,
                      genomes_matrix: csr_matrix,
                      HM: int,
                      HPC: int) -> list:
    # human_indices: indices of currently infected human agents #
    human_indices = np.where((X == HM) | (X == HPC))[0]
    selected_human = random.choice(human_indices)

    # present_genomes: genome IDs present in the selected human #
    present_genomes = genomes_matrix[:, selected_human].tocoo().row
    if len(present_genomes) <= 1:
        return list(present_genomes)

    # Sample a random subset of genomes for transmission #
    return random.sample(list(present_genomes), random.randint(1, len(present_genomes)))