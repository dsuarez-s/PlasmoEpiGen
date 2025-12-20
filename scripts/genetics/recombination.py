import numpy as np
import random
from scipy.sparse import csr_matrix
from scipy.sparse import vstack

def recombination(inoculated_genomes, parasitic_populations,
                  immature_matrix, mature_matrix,
                  generation_events, dist_loci):
    
    # Check 1: Existen genomas para inocular #
    if len(inoculated_genomes) == 0:
            raise ValueError("No genomes inoculated for recombination")
    
    # Check 2: La distribución tiene el mismo tamaño que los haplotipos #
    haplotypes_len = len(parasitic_populations[0])
    if len(dist_loci) != haplotypes_len:
        raise ValueError(f"La distribución debe tener longitud {haplotypes_len}")
    
    # Caso 1: Solo se inoculo un genoma, entonces no hay recombinación #
    if(len(inoculated_genomes) == 1):
        to_infect = inoculated_genomes.copy()
        return (parasitic_populations, immature_matrix, mature_matrix,
                generation_events, to_infect)
    
    # Caso 2: Se inoculó más de un genoma #  
    # Paso 1: Se crean las parejas de posibles recombinantes y se obtienen los esporozoitos #
    combos = [(x, y) for x in inoculated_genomes for y in inoculated_genomes]  
    num_oocy = np.random.poisson(2)
    oocy = [random.choice(combos) for _ in range(num_oocy)]
    sporozoites = list(set(oocy))

    # Paso 2: Se establece el número de eventos recombinatorios y se define el loci #
    num_events = np.random.poisson(2)    
    loci = list(np.random.choice(a=len(dist_loci), 
                                 size=min(num_events,len(dist_loci)), p=dist_loci, replace=False))

    # Paso 3: Se genera el proceso de recombinación para obtener los recombinantes #
    to_infect = []
    for pair in sporozoites:       
        # Caso 3.1: Ambos son iguales entonces no hay recombinación #
        if(pair[0] == pair[1]):
            to_infect.append(pair[0])
            continue
            
        # Caso 3.2: No son iguales entonces se producen los recombinantes #
        gen1 = list(parasitic_populations[pair[0]])
        gen2 = list(parasitic_populations[pair[1]])
        for l in set(loci):
            tmp = gen1.copy()
            gen1 = tmp[:l] + gen2[l:]
            gen2 = gen2[:l] + tmp[l:]
        child = "".join(gen1)

        existing = np.where(parasitic_populations== child)[0]

        # Caso 3.2.1  El haplotipo ha estado en algun momento en la población y ya tiene indice #
        if existing.size > 0:
            idx = int(existing[0])
            to_infect.append(idx)
            # Evento de Recombinación #
            if (immature_matrix[existing].nnz  == 0) and (mature_matrix[existing].nnz == 0):
                generation_events += 1
                
        # Caso 3.2.2  El haplotipo NO ha estado nunca en la población. Se añaden filas en ambas matrices #
        else:
            new_gen = csr_matrix((1, immature_matrix.shape[1]))
            parasitic_populations = np.append(parasitic_populations, child) # List #
            immature_matrix = vstack([immature_matrix, new_gen]).tocsr()    # Matrix #
            mature_matrix   = vstack([mature_matrix,   new_gen]).tocsr()    # Matrix #
            to_infect.append(len(parasitic_populations) - 1)
            # Evento de Recombinación Nuevo #
            generation_events += 1
                   
    return(parasitic_populations,
           immature_matrix, mature_matrix, generation_events, to_infect)