import numpy as np
import os
import heapq
from collections import defaultdict
import pickle
import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter("ignore", SparseEfficiencyWarning)

from .model_init import init_model_state
from .stochastic_engine import compute_propensities, next_time_event, variate_population
from .output_manager import save_information
from scripts.transitions.event_queue_schedule import event_queue_execution 

class MalariaEGModel:
    # Función 1: Inicializar el sistema con todos los parámetros de entrada #
    def __init__(self, epi_parameters, pop_parameters,name_folder, iteration, distribution, 
                 genomes, clone_distribution_human, clone_distribution_mosquito):
        
        init_model_state(self,epi_parameters, pop_parameters,name_folder,
                         iteration, distribution, genomes, clone_distribution_human,
                         clone_distribution_mosquito, heapq)
    
    # Función 2: Acopla todas las funciones para el desarrollo de la simulación #
    def run(self, tmax):
        
        # Paso 1: Limpiar los resultados que se habían obtenido anteriormente #
        if os.path.isfile(self.path):
            os.remove(self.path)
            
        # Paso 2: Guarda el estado inicial del sistema antes de comenzar la simulación #
        save_information(self, time_step = 0)
                
        # Paso 3. Corremos la simulación hasta alcanzar el tiempo máximo #
        t_step = 0
        while self.actual_time < tmax:
            
            nums = [(self.X == self.HS).sum(), (self.X == self.HM).sum(),
                    (self.X == self.HPC).sum(), (self.X == self.MS).sum(),
                    (self.X == self.MC).sum(), (self.X == self.MPC).sum()]
            
            # Paso 3.1: Calculamos las propensities de los eventos #
            self.propensities = compute_propensities(self)
            
            # Paso 3.2: Calculamos el tiempo, agente involucrado y tipo de transición realizada #
            self.tau, self.transitionPlayer, self.transitionType = next_time_event(self)
            
            # Paso 3.3: Actualizamos el tiempo global de la simulación #
            self.actual_time += self.tau
            
            # Paso 3.4: Revisamos si hay eventos pendientes por ejecutar y lo hacemos # 
            if(self.event_queue):
                (self.event_queue, self.immature_matrix,
                 self.mature_matrix,self.X)= event_queue_execution(event_queue = self.event_queue,
                                                                   actual_time = self.actual_time,
                                                                   immature_matrix = self.immature_matrix,
                                                                   mature_matrix = self.mature_matrix,
                                                                   X = self.X, epi_dict = self.epi)
            
            # Paso 3.5: Con base en los cambios actualizamos la población #
            variate_population(self) 
            
            # Paso 3.6: Guardamos el estado actual del sistema dado un tiempo establecido #
            while self.actual_time >= t_step and t_step <= tmax:
                save_information(self, time_step = t_step)
                t_step += 1.0