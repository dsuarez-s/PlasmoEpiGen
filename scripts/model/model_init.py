import os
from collections import defaultdict
from scripts.genetics.genome_initialization import initialize_genomes
from scripts.observables.identity_by_descent import (precompute_ibd_table, measure_ibd_relative_to_founders as measure_ibd)
from scripts.observables.nucleotide_diversity import (measure_nucleotide_diversity as measure_pi)
from scripts.observables.shannon_index import (measure_shannon_population as measure_shannon)
from scripts.observables.multiplicity_of_infection import measure_moi

def init_model_state(self, epi_parameters, pop_parameters,name_folder,
                     iteration, distribution, genomes, clone_distribution_human,
                     clone_distribution_mosquito, heapq):
    
    # Paso 1: Inicializar los diccionarios requeridos #
    self.config = {"name_folder": name_folder, "distribution": distribution}
    self.IBD_humans_mean = {}
    self.IBD_humans_median = {}
    self.IBD_mosquitoes_mean = {}
    self.IBD_mosquitoes_median = {}

    # Paso 2: Inicializar parámetros epidemiológicos #
    epi_keys = ["sigma_h", "gamma", "delta", "alpha_H", "alpha_M", "sigma_v", "beta_hv", "beta_vh"]  
    self.epi = {key: val for key, val in zip(epi_keys, epi_parameters)}   

    # Paso 3: Inicialización de contadores y estados de los agentes #
    self.actual_time = 0
    self.HS, self.HM, self.HPC = 0, 1, 2
    self.MS, self.MC, self.MPC = 3, 4, 5

    # Paso 4: Inicializar el tamaño de las poblaciones de mosquitos y humanos #  
    self.num_mos = pop_parameters["Mos"]
    self.num_hum = pop_parameters["Hum"]        

    # Paso 5: Inicializar la carpeta donde se guardarán los resultados #
    folder = self.config["name_folder"]
    os.makedirs(folder, exist_ok=True)
    fname = f'Iteration_{iteration}.txt'
    self.path = os.path.join(folder, fname)

    # Paso 6: Inicializar los eventos del sistema y los contadores de recombinación #
    self.events = ["lambda_humans", "lambda_mosquitoes", "toMS", "human_clearance"]
    self.generation_events = 0
    self.total_events = 0
    self.total_events_infect = 0
    self.infect_with_reco = 0
    
    # Paso 7: Inicializar  la lista de eventos por ocurrir #
    self.event_queue = []            
    heapq.heapify(self.event_queue)  
    
    # Paso 8: Inicializar los genomas y las matrices genéticas del sistema dada la arquitectura #
    self.genomes = genomes
    init_genomes = initialize_genomes(genomes_dictionary = genomes,
                                      HM_code = self.HM, HS_code = self.HS, HPC_code = self.HPC,
                                      MC_code = self.MC, MPC_code = self.MPC, MS_code = self.MS,
                                      clone_distribution_human = clone_distribution_human,
                                      num_mos = self.num_mos, num_hum = self.num_hum,
                                      clone_distribution_mosquito = clone_distribution_mosquito,
                                      event_queue = self.event_queue)      

    self.initial_genomes = init_genomes[0]
    self.parasitic_populations = init_genomes[0]
    self.mature_matrix = init_genomes[1]
    self.immature_matrix = init_genomes[2]  
    self.X = init_genomes[3]     