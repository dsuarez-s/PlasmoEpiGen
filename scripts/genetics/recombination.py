# Recombination and matrix update functions #
import numpy as np
import random
from scipy.sparse import csr_matrix
from scipy.sparse import vstack

def recombination(inoculated_genomes, parasitic_populations,
                  immature_matrix, mature_matrix,
                  total_events, generation_events, dist_loci):
    
    """
    Perform meiotic recombination among inoculated genomes, update genotype lists
    and sparse matrices, and return updated state plus list of genomes to infect.
    """
    
    # 0) Handle edge cases consistently #
    if not inoculated_genomes:
            raise ValueError("No genomes inoculated for recombination")
    
    # 1) If exactly one genome, no recombination needed #
    if(len(inoculated_genomes) == 1):
        to_infect = inoculated_genomes.copy()
        return (parasitic_populations, immature_matrix, mature_matrix,
                total_events, generation_events, to_infect)
    
    # 2) Perform gametocyte pairing and meiotic crossing #
    combos = [(x, y) for x in inoculated_genomes for y in inoculated_genomes]
    
    # 3) Generate oocysts (Poisson-distributed number of pairings) #
    num_oocy = np.random.poisson(2)
    oocy = [random.choice(combos) for _ in range(num_oocy)]
    # Keep unique pairs#
    sporozoites = list(set(oocy))

    # 4) Choose random crossover loci (Poisson number of events) #
    num_events = np.random.poisson(2)
    loci = list(np.random.choice(a=len(dist_loci), 
                                 size=min(num_events,len(dist_loci)), p=dist_loci, replace=False))

    to_infect = []
    # 5) For each sporozoite pair, perform recombination #
    for pair in sporozoites:
        total_events += 1
                
        # If identical genomes, no crossover #
        if(pair[0] == pair[1]):
            to_infect.append(pair[0])
            continue
        
        # Copy parental genome strings to lists #
        gen1 = list(parasitic_populations[pair[0]])
        gen2 = list(parasitic_populations[pair[1]])
                
        # Perform crossover at each locus #
        for l in set(loci):
            tmp = gen1.copy()
            gen1 = tmp[:l] + gen2[l:]
            gen2 = gen2[:l] + tmp[l:]
                
        child = "".join(gen1)
                
        # Check if child already exists in the ordered array
        existing = np.where(parasitic_populations == child)[0]
        if existing.size > 0:
            idx = int(existing[0])
            to_infect.append(idx)
        else:
            # Append child to the end of the ordered array
            parasitic_populations = np.append(parasitic_populations, child)
                
            # Extend sparse matrices by adding a new zero row #
            new_gen = csr_matrix((1, immature_matrix.shape[1]))
            immature_matrix = vstack([immature_matrix, new_gen]).tocsr()
            mature_matrix   = vstack([mature_matrix,   new_gen]).tocsr()

            new_generated_genome = parasitic_populations.size - 1
            to_infect.append(new_generated_genome)
            generation_events += 1
            
    return(parasitic_populations,
           immature_matrix, mature_matrix,total_events, generation_events, to_infect)

# -------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------- #