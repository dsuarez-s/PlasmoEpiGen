# Human classification and infection functions #
import numpy as np
from scipy.sparse import csr_matrix

############################
# Classification of States #
############################

def classification_S_M_PC(transitionPlayer, genomes_matrix):
    # Determine human state: Susceptible (S), Monoclonal (M), or Polyclonal (PC) #
    player_genomes = genomes_matrix[:, transitionPlayer].tocoo().row

    if len(player_genomes) == 0:
        return "S"
    elif len(player_genomes) == 1:
        return "M"
    else:
        return "PC"

def func_lambda_humans(transition_Player, inoculated_genomes,
                       pre_genomes_matrix, X_matrix, gamma):
    # Move human from exposed to infected by setting incubation timer #
    for i in inoculated_genomes:
        pre_genomes_matrix[i, transition_Player] = gamma
