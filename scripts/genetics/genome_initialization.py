import numpy as np  # for numerical operations
import random       # for sampling and shuffling
from scipy import sparse  # for memory-efficient sparse matrices
import heapq

# -------------------------------------------------------------------------------------------------------- #
# Assign clone counts per agent based on provided distribution #
def assign_clones(distribution, num_agents, max_haplotypes):
    
    # 1) Validate that the sum of proportions is close to 1.0 #
    total = sum(distribution.values())
    if not np.isclose(total, 1.0, atol=1e-3):
        raise ValueError(f"Sum of proportions = {total:.3f}, must be 1.0")

    # 2) Validate that requested clone count does not exceed available haplotypes #
    for n_clones, prop in distribution.items():
        if n_clones > max_haplotypes and prop > 0:
            raise ValueError(f"Cannot assign {n_clones} clones (proportion {prop}); "
                             f"only {max_haplotypes} haplotypes available.")

    # 3) Initial rounding of clone counts #
    clone_counts = []
    for clones, prop in distribution.items():
        count = int(round(prop * num_agents))
        clone_counts.extend([clones] * count)

    # 4) Ensure rounding does not exceed the total number of agents #
    if len(clone_counts) > num_agents:
        raise ValueError( f"Rounded clone counts ({len(clone_counts)}) exceed total agents ({num_agents}).")

    # 5) Fill remaining agents using weighted random sampling #
    keys, weights = zip(*distribution.items())
    while len(clone_counts) < num_agents:
        clone_counts.append(random.choices(keys, weights=weights)[0])

    random.shuffle(clone_counts)
    return np.array(clone_counts)

# -------------------------------------------------------------------------------------------------------- #
"""
Initialize parasite genome assignments for a simulated host and vector population
based on specified clone distributions and genome sequences.

Parameters:
- clone_distribution_human: dict {num_clones: proportion} specifying clone numbers for human agents.
- clone_distribution_mosquito: dict {num_clones: proportion} specifying clone numbers for mosquito agents.
- num_mos: int, number of mosquito agents.
- num_hum: int, number of human agents.
- genomes_dictionary: dict mapping genome ID to sequence (string or list of chars).
- HS_code, HM_code, HPC_code: int codes for susceptible, monoclonal, and polyclonal humans.
- MS_code, MC_code, MPC_code: int codes for susceptible, monoclonal, and polyclonal mosquitoes.
- gamma: int or float, recovery time from human infection.
- xi: int or float, parasite lifespan in mosquito salivary glands.
- event_queue: heap queue to store scheduled parasite extinction events.

Returns:
- parasitic_populations: np.array of genome sequences used in the initialization.
- mature_matrix: scipy.sparse matrix (genomes x agents) indicating assigned genomes.
- immature_matrix: empty sparse matrix with the same shape (for future infection events).
- X: np.array of agent epidemiological states (state codes for each agent).
"""

def initialize_genomes(clone_distribution_human,clone_distribution_mosquito,
                       num_mos,num_hum,genomes_dictionary,
                       HS_code, HM_code, HPC_code,
                       MS_code, MC_code, MPC_code,
                       gamma, xi, event_queue):

    # Assign the number of clones each human and mosquito agent will have #
    human_clone_counts = assign_clones(clone_distribution_human, num_hum, len(genomes_dictionary))
    mosquito_clone_counts = assign_clones(clone_distribution_mosquito, num_mos, len(genomes_dictionary))
    X_counts = list(human_clone_counts)  + list(mosquito_clone_counts)
    
    # Assign epidemiological state codes for humans based on number of clones #
    HS = (human_clone_counts == 0)*HS_code
    HM = (human_clone_counts == 1)*HM_code
    HPC = (human_clone_counts > 1)*HPC_code
    
    # Assign epidemiological state codes for mosquitoes based on number of clones #
    MS = (mosquito_clone_counts == 0)*MS_code
    MC = (mosquito_clone_counts == 1)*MC_code
    MPC = (mosquito_clone_counts > 1)*MPC_code
    
    # Combine human and mosquito states into a single array #
    human_states = HS + HM + HPC 
    mosquitoes_states = MS + MC + MPC 
    X_states = list(human_states) + list(mosquitoes_states)
    X = np.array(X_states, dtype=int)
        
    # Calculate the total number of parasite clones that need to be assigned #
    total_clones = sum(human_clone_counts) + sum(mosquito_clone_counts)
    all_genomes = list(genomes_dictionary.keys())
    num_haplotypes = len(all_genomes)
    
    # Select the genomes to assign to agents #
    if total_clones > num_haplotypes:
        selected_genomes = all_genomes
    else:
        selected_genomes = random.sample(all_genomes, total_clones)
    
    # Map genome IDs to their indices for matrix storage #
    unique_genomes = list(set(selected_genomes))
    genome_to_index = {genome: idx for idx, genome in enumerate(unique_genomes)}
    
    # Initialize sparse matrices for genome assignments in agents #
    mature_matrix = sparse.lil_matrix((len(unique_genomes), len(X)), dtype=int)
    immature_matrix = sparse.lil_matrix((len(unique_genomes), len(X)), dtype=int)

   # Calculate total number of clones to assign and number of available genomes (usually defined above)
    num_genomes = len(selected_genomes)

    # CASE 1: Enough genomes for all clones, assign globally unique haplotypes (no repetitions at all) #
    if total_clones <= num_genomes:
        for position_agent in range(len(X_counts)):
            num_clones = X_counts[position_agent]
            agent_state = X[position_agent]
            if num_clones > 0:
                # Randomly sample haplotypes for this agent from the global pool
                haplotypes = random.sample(selected_genomes, num_clones)
                for genome in haplotypes:
                    genome_ID = genome_to_index[genome]
                    mature_matrix[genome_ID, position_agent] = 1
                    # Remove from the global pool so it can't be assigned again #
                    selected_genomes.remove(genome)  

                    # Schedule a death event for each assigned haplotype
                    type_event = "Death"
                    agent = position_agent
                    # Choose event time based on host type #
                    if agent_state > HPC_code:
                        # Mosquito: parasite lifespan in mosquito salivary glands #
                        t_event = xi  
                    else:
                        # Human: recovery time #
                        t_event = gamma  
                    heapq.heappush(event_queue, (t_event, type_event, genome_ID, agent))
    # CASE 2: Not enough genomes for global uniqueness, only guarantee unique clones per agent #
    else:
        for position_agent in range(len(X_counts)):
            num_clones = X_counts[position_agent]
            agent_state = X[position_agent]
            if num_clones > 0:
                # Assign unique haplotypes per agent, but allow reuse between agents #
                haplotypes = random.sample(selected_genomes, num_clones)
                for genome in haplotypes:
                    row = genome_to_index[genome]
                    mature_matrix[row, position_agent] = 1

                    # Schedule a death event for each assigned haplotype #
                    type_event = "Death"
                    agent = position_agent
                    # Choose event time based on host type #
                    if agent_state > HPC_code:
                        # Mosquito: parasite lifespan in mosquito salivary glands #
                        t_event = xi  
                    else:
                        # Human: recovery time #
                        t_event = gamma  
                    heapq.heappush(event_queue, (t_event, type_event, row, agent))

    # Convert genome sequences to a NumPy array #
    parasitic_list = []
    for g in unique_genomes:
        seq = genomes_dictionary[g]
        if isinstance(seq, str):
            parasitic_list.append(seq)
        else:
            parasitic_list.append("".join(seq))

    parasitic_populations = np.array(parasitic_list)
    
    # Return genome sequences, the infection matrices, and state vector #
    return parasitic_populations, mature_matrix.tocsr(), immature_matrix.tocsr(), X 
   
# -------------------------------------------------------------------------------------------------------- #