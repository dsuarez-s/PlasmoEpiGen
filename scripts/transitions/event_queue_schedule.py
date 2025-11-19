import heapq
from scripts.helpers.state_inspectors import classification_S_M_PC
import numpy as np 
 
def event_queue_execution(event_queue, actual_time, immature_matrix, mature_matrix, X, epi_dict):
    
    while event_queue and event_queue[0][0] < actual_time:
        
        # Paso 1: LLamamos al evento más reciente (Gametocytes Maturation o Sporozoites Maturation) # 
        next_in_queue_time, evt_type, genome_ID, agent = heapq.heappop(event_queue)
        count = immature_matrix[genome_ID, agent] # Extraemos el conteo de dicho haplotipo en un agente #
        
        # Check 1: Si es cero o menor entonces algo esta mal porque esta entrando como infectado #
        if (count <= 0):
            print("DEBUG ERROR:")
            print(f"Event: {evt_type} | genome_ID={genome_ID} | agent={agent}")
            print(f"Actual time: {actual_time}")
            print(f"In event queue time: {next_in_queue_time}")
            print(f"Immature count: {immature_matrix[genome_ID, agent]}")
            print(f"Mature count: {mature_matrix[genome_ID, agent]}")
            print(f"Sum immature row: {immature_matrix[genome_ID].sum()}")
            print(f"Sum mature row: {mature_matrix[genome_ID].sum()}")
            raise RuntimeError(f"Conteo de cero para '{evt_type}' en genome {genome_ID} agent {agent}")
        
        # Paso 2: El parásito madura y pasa de la matriz de inmaduros a maduros #
        immature_matrix[genome_ID, agent] -= 1
        mature_matrix  [genome_ID, agent] += 1
        X = classification_S_M_PC(transitionPlayer=agent, X_matrix=X, mature_matrix=mature_matrix)

    # Check 2: La matrix de inmaduros llegó a ser negativa #
    if (immature_matrix.data < 0).any():
        raise ValueError("Immature_matrix contiene valores negativos, lo cual es inválido.")

    # Paso 3: Eliminamos los nuevos ceros de la matriz #
    immature_matrix.eliminate_zeros()
    
    return (event_queue, immature_matrix, mature_matrix, X)         