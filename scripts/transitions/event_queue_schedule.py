import heapq
from scripts.helpers.state_inspectors import classification_S_M_PC,update_matrices

def event_queue_execution(event_queue, actual_time,
                          immature_matrix, mature_matrix, X, epi_dict, p_populations):
    
    while event_queue and event_queue[0][0] < actual_time:
        
        next_in_queue_time, evt_type, genome_ID, agent = heapq.heappop(event_queue)
        
        if evt_type in ("Gametocytes Maturation","Sporozoites Maturation"):
            immature_matrix[genome_ID, agent] -= 1
            mature_matrix  [genome_ID, agent] += 1

            if(evt_type == "Gametocytes Maturation"):
                t_next = next_in_queue_time + epi_dict["gamma"]
            else:
                t_next = next_in_queue_time + epi_dict["xi"]
            
            heapq.heappush(event_queue,(t_next, "Death", genome_ID, agent))

            X = classification_S_M_PC(transitionPlayer=agent,
                                      X_matrix=X,
                                      mature_matrix=mature_matrix)

        elif evt_type == "Death":
            # Quita 1 clon de mature
            mature_matrix[genome_ID, agent] -= 1

            X = classification_S_M_PC(transitionPlayer = agent,
                                      X_matrix = X,
                                      mature_matrix = mature_matrix)
        
    # Verificación de valores negativos en las matrices #
    if (immature_matrix.data < 0).any():
        raise ValueError("immature_matrix contiene valores negativos, lo cual es inválido.")

    if (mature_matrix.data < 0).any():
        raise ValueError("mature_matrix contiene valores negativos, lo cual es inválido.")

    #  Podar haplotipos extintos tras cada muerte de parásito #
    pop_matrix = update_matrices(mature_matrix = mature_matrix,
                                 immature_matrix = immature_matrix,
                                 parasitic_populations = p_populations)

    parasitic_populations = pop_matrix[0]
    mature_matrix = pop_matrix[1]
    immature_matrix = pop_matrix[2]
    
    return (event_queue, immature_matrix, mature_matrix, X,parasitic_populations)         