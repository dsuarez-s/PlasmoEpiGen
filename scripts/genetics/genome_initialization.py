import numpy as np  # for numerical operations
import random       # for sampling and shuffling
from scipy import sparse  # for memory-efficient sparse matrices

# -------------------------------------------------------------------------------------------------------- #
# Assign clone counts per agent based on provided distribution #
def assign_clones(distribution, num_agents):
    clone_counts = []

    # Validate that the sum of proportions is approximately 1.0 #
    total_proportion = sum(distribution.values())
    if not np.isclose(total_proportion, 1.0, atol=1e-3):
        raise ValueError(f"Clone distribution proportions must sum to 1.0, but got {total_proportion:.4f}.")

    # Calculate how many agents should receive each clone count #
    for clones, proportion in distribution.items():
        count = int(round(proportion * num_agents))
        clone_counts.extend([clones] * count)

    # Raise an error if rounding over-assigns #
    if len(clone_counts) > num_agents:
        raise ValueError("The total assigned proportion exceeds 1.0 after rounding. Please adjust the distribution.")

    # Fill remaining agents using weighted random sampling based on the original distribution #
    while len(clone_counts) < num_agents:
        clone_counts.append(
            random.choices(list(distribution.keys()), weights=distribution.values())[0]
        )

    random.shuffle(clone_counts)  # Randomize the assignment order
    return clone_counts

# -------------------------------------------------------------------------------------------------------- #
"""
Assign parasite genomes to infected agents based on clone distributions.

Parameters:
- X: Array of agent state codes.
- gamma: Recovery time of a human from an infection.
- xi: Lifespan of parasites in mosquito salivary glands.
- genomes_dictionary: Mapping from genome ID to its sequence.
- HM_code, HPC_code, MC_code, MPC_code: State codes for different infection types.
- clone_distribution_human: Dictionary {num_clones: proportion} for human agents.
- clone_distribution_mosquito: Dictionary {num_clones: proportion} for mosquito agents.

Returns:
- parasitic_populations: Array of genome sequences.
- genomes_matrix: Sparse matrix (genomes x agents) with infection timers.
- pre_genomes_matrix: Empty matrix with the same shape (used later in the simulation).
"""
def initialize_genomes(X, gamma, xi, genomes_dictionary,
                       HM_code, HPC_code, MC_code, MPC_code,
                       clone_distribution_human,
                       clone_distribution_mosquito):

    # Identify infected human and mosquito agents #
    monoclonal_humans = np.where(X == HM_code)[0]
    polyclonal_humans = np.where(X == HPC_code)[0]
    monoclonal_mosquitoes = np.where(X == MC_code)[0]
    polyclonal_mosquitoes = np.where(X == MPC_code)[0]

    infected_humans = np.concatenate([monoclonal_humans, polyclonal_humans])
    infected_mosquitoes = np.concatenate([monoclonal_mosquitoes, polyclonal_mosquitoes])

    # Determine number of clones each agent will carry #
    human_clone_counts = assign_clones(clone_distribution_human, len(infected_humans))
    mosquito_clone_counts = assign_clones(clone_distribution_mosquito, len(infected_mosquitoes))

    # Total number of clones to assign #
    total_clones = sum(human_clone_counts) + sum(mosquito_clone_counts)

    # Ensure enough genomes are available for all clones #
    all_genomes = list(genomes_dictionary.keys())
    while len(all_genomes) < total_clones:
        all_genomes.extend(list(genomes_dictionary.keys()))
    selected_genomes = random.sample(all_genomes, total_clones)

    # Map unique genome IDs to row indices in the matrix #
    unique_genomes = list(set(selected_genomes))
    genome_to_index = {genome: idx for idx, genome in enumerate(unique_genomes)}

    # Create sparse matrices to hold genome-agent associations #
    num_genomes = len(unique_genomes)
    genomes_matrix = sparse.lil_matrix((num_genomes, len(X)), dtype=int)
    pre_genomes_matrix = sparse.lil_matrix((num_genomes, len(X)), dtype=int)

    # Assign genomes to infected humans #
    pos = 0
    for agent, num_clones in zip(infected_humans, human_clone_counts):
        for _ in range(num_clones):
            genome = selected_genomes[pos]
            row = genome_to_index[genome]
            genomes_matrix[row, agent] = gamma
            pos += 1

    # Assign genomes to infected mosquitoes #
    for agent, num_clones in zip(infected_mosquitoes, mosquito_clone_counts):
        for _ in range(num_clones):
            genome = selected_genomes[pos]
            row = genome_to_index[genome]
            genomes_matrix[row, agent] = xi
            pos += 1

    # Convert genome sequences to a NumPy array #
    parasitic_populations = np.array([
        "".join(genomes_dictionary[g]) if not isinstance(genomes_dictionary[g], str)
        else genomes_dictionary[g]
        for g in unique_genomes
    ])

    return parasitic_populations, genomes_matrix.tocsr(), pre_genomes_matrix.tocsr()

# -------------------------------------------------------------------------------------------------------- #