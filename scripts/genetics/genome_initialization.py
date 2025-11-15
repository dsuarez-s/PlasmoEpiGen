import numpy as np 
import random     
from scipy import sparse 

# -------------------------------------------------------------------------------- #
def assign_clones(distribution, num_agents, max_haplotypes):
    
    # Check 1: Verificar que la distribución de clones es correcta #
    total = sum(distribution.values())
    if not np.isclose(total, 1.0, atol=1e-3):
        raise ValueError(f"Sum of proportions = {total:.3f}, must be 1.0")

    # Check 2: Verificar que cantidad de haplotipos sea suficiente #
    for n_clones, prop in distribution.items():
        if n_clones > max_haplotypes and prop > 0:
            raise ValueError(f"Cannot assign {n_clones} clones (proportion {prop}); "
                             f"only {max_haplotypes} haplotypes available.")

    # Paso 1: Asignar en una lista el numero de clones requeridos para los agentes #
    clone_counts = []
    for clones, prop in distribution.items():
        count = int(round(prop * num_agents))
        clone_counts.extend([clones] * count)

    # Check 3: Revisar que el redondeo no genere más agentes #
    if len(clone_counts) > num_agents :
        raise ValueError( f"Rounded counts ({len(clone_counts)}) differs from total agents ({num_agents}).")

    # Paso 2: Si faltó algún agente se asigna uno de forma aleatoria #
    keys, weights = zip(*distribution.items())
    while len(clone_counts) < num_agents:
        clone_counts.append(random.choices(keys, weights=weights)[0])

    random.shuffle(clone_counts)
    return np.array(clone_counts)

# -------------------------------------------------------------------------------- #
def initialize_genomes(clone_distribution_human,clone_distribution_mosquito,
                       num_mos,num_hum,genomes_dictionary,
                       HS_code, HM_code, HPC_code,
                       MS_code, MC_code, MPC_code,
                       event_queue):

    # Paso 1: Asignar el numero de clones que los humanos y mosquitos tendrán #
    human_clone_counts = assign_clones(clone_distribution_human, num_hum, len(genomes_dictionary))
    mosquito_clone_counts = assign_clones(clone_distribution_mosquito, num_mos, len(genomes_dictionary))
    X_counts = list(human_clone_counts)  + list(mosquito_clone_counts)
    
    # Paso 2: Crear el vector que contiene los estados iniciales de los agentes #
    HS = (human_clone_counts == 0)*HS_code;    HM = (human_clone_counts == 1)*HM_code;
    HPC = (human_clone_counts > 1)*HPC_code;   MS = (mosquito_clone_counts == 0)*MS_code;
    MC = (mosquito_clone_counts == 1)*MC_code; MPC = (mosquito_clone_counts > 1)*MPC_code;
    human_states = HS + HM + HPC; mosquitoes_states = MS + MC + MPC;
    X_states = list(human_states) + list(mosquitoes_states)
    X = np.array(X_states, dtype=int)
    
    # Paso 3: Calcular el numéro total de clones necsitados por la población #
    total_clones = sum(human_clone_counts) + sum(mosquito_clone_counts);
    all_genomes = list(genomes_dictionary.keys()); num_haplotypes = len(all_genomes);
    if total_clones > num_haplotypes:
        selected_genomes = all_genomes
    else:
        selected_genomes = random.sample(all_genomes, total_clones)
    num_genomes = len(selected_genomes)
    
    # Paso 4: Inicializar las matrices genéticas mature e immature #
    unique_genomes = list(set(selected_genomes));
    genome_to_index = {genome: idx for idx, genome in enumerate(unique_genomes)};
    mature_matrix = sparse.lil_matrix((len(unique_genomes), len(X)), dtype=int)
    immature_matrix = sparse.lil_matrix((len(unique_genomes), len(X)), dtype=int)
    
    # Paso 5: Asignar los clones a la población #
    # Caso 5.1: El número de clones es suficiente # 
    if total_clones <= num_genomes:
        for position_agent in range(len(X_counts)):
            num_clones = X_counts[position_agent] # Clones requeridos #
            agent_state = X[position_agent] # Estado del agente #
            if num_clones > 0:
                haplotypes = random.sample(selected_genomes, num_clones)
                for genome in haplotypes:
                    genome_ID = genome_to_index[genome]
                    mature_matrix[genome_ID, position_agent] = 1 # Asignación a la matrix madura #
                    selected_genomes.remove(genome) # Evitar repetición eliminando el haplotipo usado #

    # Caso 5.2: El número de genomas no es suficiente y se distribuirán equitativamente en la población # 
    else:
        for position_agent in range(len(X_counts)):
            num_clones = X_counts[position_agent] # Clones requeridos #
            agent_state = X[position_agent] # Estado del agente #
            if num_clones > 0:
                haplotypes = random.sample(selected_genomes, num_clones)
                for genome in haplotypes:
                    genome_ID = genome_to_index[genome]
                    mature_matrix[genome_ID, position_agent] = 1

    # Paso 6: Convertir la lista de parásitos en un array #
    parasitic_list = [];
    for genome in unique_genomes:
        sequence = genomes_dictionary[genome]
        if isinstance(sequence, str):
            parasitic_list.append(sequence)
        else:
            parasitic_list.append("".join(sequence))

    parasitic_populations = np.array(parasitic_list);
    
    return parasitic_populations, mature_matrix.tocsr(), immature_matrix.tocsr(), X 
   
# -------------------------------------------------------------------------------------------------------- #