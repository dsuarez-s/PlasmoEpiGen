# Recombination and matrix update functions #
import numpy as np
import random
from scipy.sparse import csr_matrix

def recombination(inoculated_genomes, parasitic_populations,
                  pre_genomes_matrix, genomes_matrix,
                  total_events, generation_events, distribution):
    # Perform gametocyte pairing and meiotic crossing #
    combos = [(x, y) for x in inoculated_genomes for y in inoculated_genomes]
    num_oocy = np.random.poisson(2)
    oocy = [random.choice(combos) for _ in range(num_oocy)]
    sporozoites = list(set(oocy))

    # Determine crossover loci #
    num_events = np.random.poisson(2)
    loci = list(np.random.choice(np.arange(20), size=num_events,
                                 p=distribution, replace=False))

    to_infect = []
    for pair in sporozoites:
        total_events += 1
        if pair[0] != pair[1]:
            # Crossover between two different genomes #
            gen1 = list(parasitic_populations[pair[0]])
            gen2 = list(parasitic_populations[pair[1]])
            for l in set(loci):
                tmp = gen1
                gen1 = tmp[:l] + gen2[l:]
                gen2 = gen2[:l] + tmp[l:]
            child = "".join(gen1)
            if child not in parasitic_populations:
                # Add new genome #
                parasitic_populations = np.append(
                    parasitic_populations, child)
                rows = pre_genomes_matrix.shape[0]
                pre_genomes_matrix._shape = (rows + 1,
                                             pre_genomes_matrix.shape[1])
                genomes_matrix._shape = (rows + 1,
                                         genomes_matrix.shape[1])
                pre_genomes_matrix.indptr = np.hstack(
                    (pre_genomes_matrix.indptr, pre_genomes_matrix.indptr[-1]))
                genomes_matrix.indptr = np.hstack(
                    (genomes_matrix.indptr, genomes_matrix.indptr[-1]))
                to_infect.append(len(parasitic_populations) - 1)
                generation_events += 1
            else:
                idx = np.where(parasitic_populations == child)[0][0]
                to_infect.append(idx)
        else:
            to_infect.append(pair[0])
    return [parasitic_populations,
            pre_genomes_matrix,
            genomes_matrix,
            total_events,
            generation_events,
            to_infect]

def update_matrices(genomes_matrix, pre_genomes_matrix,
                    parasitic_populations):
    # Remove extinct haplotypes from matrices #
    combined = genomes_matrix + pre_genomes_matrix
    alive = np.asarray(combined.sum(axis=1)).squeeze() != 0
    return [
        parasitic_populations[alive],
        genomes_matrix[alive],
        pre_genomes_matrix[alive]
    ]

# States: HS=0, HM=1, HPC=2, MS=3, MC=4, MPC=5 #
HS, HM, HPC = 0, 1, 2
MS, MC, MPC = 3, 4, 5

def classification_S_M_PC(transitionPlayer, X_matrix,
                          humans_positions, genomes_matrix):
    # Update agent state based on genome presence #
    player_genomes = genomes_matrix[:, transitionPlayer].tocoo().row
    if len(player_genomes) == 0:
        X_matrix[transitionPlayer] = HS if transitionPlayer in humans_positions else MS
    elif len(player_genomes) == 1:
        X_matrix[transitionPlayer] = HM if transitionPlayer in humans_positions else MC
    else:
        X_matrix[transitionPlayer] = HPC if transitionPlayer in humans_positions else MPC
