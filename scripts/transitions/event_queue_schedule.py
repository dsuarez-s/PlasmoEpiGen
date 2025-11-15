import heapq
from scripts.helpers.state_inspectors import classification_S_M_PC
import numpy as np 
 
def event_queue_execution(event_queue, actual_time, immature_matrix, mature_matrix, X, epi_dict):
    
    while event_queue and event_queue[0][0] < actual_time:

        next_in_queue_time, evt_type, genome_ID, agent = heapq.heappop(event_queue)
            
        if evt_type in ("Gametocytes Maturation", "Sporozoites Maturation"):
            count = immature_matrix[genome_ID, agent]
        else:
            count = mature_matrix[genome_ID, agent]

        if (count == 0):
            print("DEBUG ERROR:")
            print(f"Event: {evt_type} | genome_ID={genome_ID} | agent={agent}")
            print(f"Actual time: {actual_time}")
            print(f"In event queue time: {next_in_queue_time}")
            print(f"Immature count: {immature_matrix[genome_ID, agent]}")
            print(f"Mature count: {mature_matrix[genome_ID, agent]}")
            print(f"Sum immature row: {immature_matrix[genome_ID].sum()}")
            print(f"Sum mature row: {mature_matrix[genome_ID].sum()}")
            raise RuntimeError(f"Conteo de cero para '{evt_type}' en genome {genome_ID} agent {agent}")

        elif count < 0:
            raise RuntimeError(f"Conteo negativo para '{evt_type}' en genome {genome_ID} agent {agent}")
        # --------------------------------------------------------------- #
        if evt_type in ("Gametocytes Maturation","Sporozoites Maturation"):
            immature_matrix[genome_ID, agent] -= 1
            mature_matrix  [genome_ID, agent] += 1

            X = classification_S_M_PC(transitionPlayer=agent,
                                      X_matrix=X,
                                      mature_matrix=mature_matrix)

    
    # Verificación de valores negativos en las matrices #
    if (immature_matrix.data < 0).any():
        raise ValueError("Immature_matrix contiene valores negativos, lo cual es inválido.")

    if (mature_matrix.data < 0).any():
        raise ValueError("Mature_matrix contiene valores negativos, lo cual es inválido.")
        
    immature_matrix.eliminate_zeros()
    mature_matrix.eliminate_zeros()
    
    return (event_queue, immature_matrix, mature_matrix, X)         