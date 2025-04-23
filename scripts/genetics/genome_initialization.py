import numpy as np
import random
from scipy import sparse

def initialize_genomes(X, gamma, xi, genomes, HM, MC):
    current_HM = np.where(X == HM)[0]
    current_MC = np.where(X == MC)[0]
    tot_init_inf = len(current_HM) + len(current_MC)
    tot_list_inf = np.concatenate([current_HM, current_MC])

    if len(genomes) < tot_init_inf:
        reps = tot_init_inf // len(genomes) + 1
        gen_f = list(genomes.keys()) * reps
    else:
        gen_f = list(genomes.keys())

    h_selected = random.sample(gen_f, tot_init_inf)

    unique = set(h_selected)
    parasitic_populations = np.array([])
    size = len(unique)
    N = len(X)
    genomes_matrix = sparse.csr_matrix((size, N), dtype=int)
    pre_genomes_matrix = sparse.csr_matrix((size, N), dtype=int)

    mapping = {}
    for idx, val in enumerate(unique):
        mapping[val] = idx
        seq = genomes[val]
        parasitic_populations = np.append(
            parasitic_populations,
            "".join(seq) if not isinstance(seq, str) else seq)

    for i, h in enumerate(h_selected):
        pos = mapping[h]
        agent = tot_list_inf[i]
        timer = gamma if agent in current_HM else xi
        genomes_matrix[pos, agent] = timer

    return parasitic_populations, genomes_matrix, pre_genomes_matrix