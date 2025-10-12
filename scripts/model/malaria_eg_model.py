# -------------------------------- #
# Part 1: Imports and Dependencies #
# -------------------------------- #
import numpy as np
import os
import heapq
from collections import defaultdict
import pickle

# imports relativos desde subpaquetes dentro del paquete `model`
from .model_init import init_model_state
from .stochastic_engine import compute_propensities, next_time_event, variate_population
from .output_manager import save_information

# imports globales # 
from scripts.transitions.event_queue_schedule import event_queue_execution 

class MalariaEGModel:
    def __init__(self, epi_parameters, pop_parameters,
                 name_folder, iteration, distribution, genomes,
                 clone_distribution_human, clone_distribution_mosquito):
        
        # Model Initialization (__init__) #
        init_model_state(self,epi_parameters, pop_parameters,name_folder,
                         iteration, distribution, genomes, clone_distribution_human,
                         clone_distribution_mosquito, heapq=heapq)
    
    # Main Simulation Loop #
    def run(self, tmax):
        # 1. Limpiar resultados previos si existe el archivo #
        if os.path.isfile(self.path):
            os.remove(self.path)
            
        # 2. Guardar estado inicial #
        save_information(self, time_step = 0)
                
        # 3. Simulacion #
        t_step = 1
        while self.actual_time < tmax:
            self.propensities = compute_propensities(self)
            self.tau, self.transitionPlayer, self.transitionType = next_time_event(self)
            self.actual_time += self.tau

            if(self.event_queue):
                (self.event_queue, self.immature_matrix, self.mature_matrix,self.X,
                 self.parasitic_populations)= event_queue_execution(event_queue = self.event_queue,
                                                                    actual_time = self.actual_time,
                                                                    immature_matrix = self.immature_matrix,
                                                                    mature_matrix = self.mature_matrix,
                                                                    X = self.X, epi_dict = self.epi,
                                                                    p_populations = self.parasitic_populations)
                
            variate_population(self) 
            while self.actual_time >= t_step and t_step <= tmax:
                save_information(self, time_step = t_step)
                t_step += 1

        
        final_results = {"humans_median" : self.observables_humans_median,
                         "humans_mean" : self.observables_humans_mean,
                         "mosquitoes_median" : self.observables_mosquitoes_median,
                         "mosquitoes_mean" : self.observables_mosquitoes_mean}

        # Specify the filename for the pickle file #
        file_dict =  os.path.join(self.config["name_folder"],  "tmp_results_" + str(self.config["iteration"]) + ".pkl")

        # Open the file in binary write mode ('wb')
        with open(file_dict, 'wb') as file:
            # Dump the dictionary into the file
            pickle.dump(final_results, file)       