# Main Malaria Epidemiological and Genetic Model #

# -------------------------------- #
# Part 1: Imports and Dependencies #
# -------------------------------- #
import numpy as np
import random
import os
from scipy import sparse
import heapq

# Helpers para populación y genomas #
from scripts.helpers.population_initialization import initialize_population_counts
from scripts.genetics.genome_initialization    import initialize_genomes

# Cálculo de tasas #
from scripts.transitions.calculate_forces import compute_propensities

# Funciones de transición de estado #
from scripts.transitions.mosquitoes_events import func_toMS, mosquito_to_human
from scripts.transitions.humans_events    import human_to_mosquito

# Recombination y limpieza de haplotipos #
from scripts.genetics.recombination import recombination, update_matrices, classification_S_M_PC

"""
sigma_h:   Number of bites per mosquito
gamma:     Recovery time of a human from infection
delta:     Lifespan of mosquitoes
alpha_H:   Maturation time of gametocytes in humans
alpha_M:   Maturation time of sporozoites in mosquitoes
sigma_v:   Rate of gonotrophic cycle (mosquito feeding cycle)
beta_hv:   Probability of transmission from mosquito to human
beta_vh:   Probability of transmission from human to mosquito
xi:        Lifespan of parasites in mosquito salivary glands
"""

# --------------------------------------- #
# Part 2: Model Initialization (__init__) #
# --------------------------------------- #
class MalariaEGModel:
    def __init__(self, epi_parameters, pop_parameters,
                 name_folder, size_pool, iteration, distribution, genomes,
                 clone_distribution_human, clone_distribution_mosquito):
        
        self.event_queue = []            
        heapq.heapify(self.event_queue)  
        
        # Initialize model parameters #
        self.config = {"name_folder": name_folder,"size_pool": size_pool,
                       "iteration": iteration, "distribution": distribution}
        
        # Initialize epidemiological  parameters #
        epi_keys = ["sigma_h", "gamma", "delta", "alpha_H", "alpha_M", "sigma_v", "beta_hv", "beta_vh", "xi"]  
        self.epi = {key: val for key, val in zip(epi_keys, epi_parameters)}   

        # Time and state counters #
        self.actual_time = 0
        self.HS, self.HM, self.HPC = 0, 1, 2
        self.MS, self.MC, self.MPC = 3, 4, 5

        # Initiliaze Populations #  
        
        self.num_mos = pop_parameters["Mos"]
        self.num_hum = pop_parameters["Hum"]        

        # Event list and counters #
        self.events = ["lambda_humans", "lambda_mosquitoes", "toMS"]
        self.generation_events = 0
        self.total_events = 0
        
        # Genetic initialization #
        init_genomes = initialize_genomes(gamma = self.epi["gamma"] , xi = self.epi["xi"],
                                          genomes_dictionary = genomes,
                                          HM_code = self.HM, HPC_code = self.HPC,
                                          MC_code = self.MC, MPC_code = self.MPC,
                                          clone_distribution_human = clone_distribution_human,
                                          clone_distribution_mosquito = clone_distribution_mosquito,
                                          event_queue = self.event_queue)
        
                   
        self.parasitic_populations = init_genomes[0]
        self.mature_matrix = init_genomes[1]
        self.immature_matrix = init_genomes[2]  
        self.X = init_genomes[3]  
    
    # ------------------------------ #
    # Part 3: Computing Propensities #
    # ------------------------------ #
    
    def calculate_forces(self):
        self.propensities = compute_propensities(X = self.X, Hum_Pop = self.num_hum,
                                                 Mos_Pop = self.num_mos,
                                                 sigma_v = self.epi["sigma_v"], sigma_h = self.epi["sigma_h"],
                                                 beta_hv = self.epi["beta_hv"], beta_vh = self.epi["beta_vh"],
                                                 delta = self.epi["delta"])
    
    # ---------------------------- #
    # Part 4: Selecting Next Event #
    # ---------------------------- #

    def next_time_event(self):
        # Determine next event and time increment #
        total = self.propensities.sum()
        cum = np.cumsum(self.propensities)
        r = np.random.rand()
        self.tau = np.random.exponential(1 / total)
        idx = np.searchsorted(cum, r * total)
        self.transitionPlayer = idx % len(self.X)
        self.transitionType = self.events[idx // len(self.X)]

    # ---------------------------------- #
    # Part 5: Applying State Transitions #
    # ---------------------------------- #
        
    def variate_population(self):
        # Apply state transitions based on event type #
        if self.transitionType == "lambda_humans":
            inoc = mosquito_to_human(X = self.X, 
                                     mature_matrix = self.mature_matrix,
                                     MC_code = self.MC, MPC_code = self.MPC)
            
            for g in inoc:
                if self.immature_matrix[g, self.transitionPlayer] == 0:
                    self.immature_matrix[g, self.transitionPlayer] += 1
                    
                    # Schedule a death event for each assigned haplotype
                    genome_ID = g
                    t_event = self.alpha_H + self.actual_time 
                    type_event = "Gametocytes Maturation"
                    agent = self.transitionPlayer
                    heapq.heappush(self.event_queue, (t_event, type_event, genome_ID, agent))

                            
            self.X = classification_S_M_PC(transitionPlayer=self.transitionPlayer,
                                           X_matrix=self.X,
                                           mature_matrix=self.mature_matrix)

        elif self.transitionType == "lambda_mosquitoes":
            inoc = human_to_mosquito(X = self.X,
                                     mature_matrix=self.mature_matrix,
                                     HM_code=self.HM, HPC_code =self.HPC)
                       
            result = recombination(inoculated_genomes=inoc, 
                                   parasitic_populations=self.parasitic_populations,
                                   immature_matrix=self.immature_matrix,
                                   mature_matrix=self.mature_matrix,
                                   total_events=self.total_events,
                                   generation_events=self.generation_events,
                                   dist_loci=self.config["distribution"])
                
            self.parasitic_populations = result[0]
            self.immature_matrix = result[1]
            self.mature_matrix = result[2]
            self.total_events = result[3]
            self.generation_events = result[4]
            selected = result[5]
            
            for g in selected:
                if(self.immature_matrix[g, self.transitionPlayer] == 0):
                    self.immature_matrix[g, self.transitionPlayer] += 1
                    
                    # Schedule a death event for each assigned haplotype #
                    genome_ID = g
                    t_event = self.alpha_M + self.actual_time 
                    type_event = "Sporozoites Maturation"
                    agent = self.transitionPlayer
                    heapq.heappush(self.event_queue, (t_event, type_event, genome_ID, agent))
                    
            self.X = classification_S_M_PC(transitionPlayer=self.transitionPlayer,
                                           X_matrix=self.X,
                                           mature_matrix=self.mature_matrix)
        else:
            # Mosquito death/reset to susceptible #
            matrices = func_toMS(transitionPlayer = self.transitionPlayer,
                                 X_matrix = self.X,
                                 immature_matrix = self.immature_matrix,
                                 mature_matrix = self.mature_matrix,
                                 MS_code = self.MS)
            
            self.X = matrices[0]
            self.mature_matrix = matrices[1]
            self.immature_matrix = matrices[2]
    
            pop_matrix = update_matrices(mature_matrix = self.mature_matrix,
                                         immature_matrix = self.immature_matrix,
                                         parasitic_populations = self.parasitic_populations)

            self.parasitic_populations = pop_matrix[0]
            self.mature_matrix = pop_matrix[1]
            self.immature_matrix = pop_matrix[2]
    
            self.X = classification_S_M_PC(transitionPlayer=self.transitionPlayer,
                                           X_matrix=self.X,
                                           mature_matrix=self.mature_matrix)
        
    # -------------------------------- #
    # Part 6: Saving Simulation Output #
    # -------------------------------- #
    def save_information(self, time_step: int, ratio_reco: float, num_haplotypes: int):
        """
        Guarda en un archivo de texto la evolución del sistema.
        - Escribe un encabezado si el archivo no existía.
        - Agrega una línea por cada llamada con:
          time_step;HS;HM;HPC;MS;MC;MPC;ratio_reco;num_haplotypes
        """
        # 1) Ruta de la carpeta y del archivo
        folder = self.config["name_folder"]
        os.makedirs(folder, exist_ok=True)

        fname = f'BR_{self.epi["beta_hv"]}_IGD_{self.config["size_pool"]}_{self.config["iteration"]}.txt'
        path = os.path.join(folder, fname)

        # 2) Encabezado (solo si es la primera vez)
        header = "time;HS;HM;HPC;MS;MC;MPC;ratio_reco;num_haplotypes"
        if not os.path.isfile(path):
            with open(path, "w") as f:
                f.write(header + "\n")

        # 3) Conteo de cada estado
        nums = [ (self.X == self.HS).sum(), (self.X == self.HM).sum(),
                (self.X == self.HPC).sum(), (self.X == self.MS).sum(),
                (self.X == self.MC).sum(), (self.X == self.MPC).sum()]

        # 4) Línea a registrar
        row = ";".join([str(time_step)] + [str(n) for n in nums] + [f"{ratio_reco}", f"{num_haplotypes}"])

        # 5) Escritura en modo append
        with open(path, "a") as f:
            f.write(row + "\n")
            
    # ---------------------------- #
    # Part 7: Main Simulation Loop #
    # ---------------------------- #
    def run(self, tmax):
        # Run the full simulation up to tmax #
        #self.save_information(0, 0, self.genomes_matrix.shape[0])
        time_step = 1
        while self.actual_time < tmax:    
            self.actual_time += self.tau  
            while self.event_queue and self.event_queue[0][0] < self.actual_time:

                # 2) Comprueba el próximo evento de parásito (si existe) #
                if(self.event_queue):
                    next_in_queue_time, evt_type, genome_ID, agent = self.event_queue[0]
                else:
                    next_in_queue_time = np.inf

                # 3) Decide qué evento ejecutar #
                if(next_in_queue_time < self.actual_time and next_in_queue_time <= tmax):
                    # — Evento de parásito “a tiempo” #
                    heapq.heappop(self.event_queue)

                    if(evt_type == "Gametocytes Maturation"):
                        # Mueve 1 clon de immature -> mature #
                        self.immature_matrix[genome_ID, agent] -= 1
                        self.mature_matrix  [genome_ID, agent] += 1

                        # Encola su muerte futura #
                        t_next = self.actual_time + self.epi["gamma"]
                        heapq.heappush(self.event_queue,(t_next, "Death", genome_ID, agent))
                        
                        self.X = classification_S_M_PC(transitionPlayer=agent,
                                                       X_matrix=self.X,
                                                       mature_matrix=self.mature_matrix)

                    elif(evt_type == "Sporozoites Maturation"):
                        # Mueve 1 clon de immature -> mature #
                        self.immature_matrix[genome_ID, agent] -= 1
                        self.mature_matrix  [genome_ID, agent] += 1

                        # Encola su muerte futura #
                        t_next = self.actual_time + self.epi["xi"]
                        heapq.heappush(self.event_queue,(t_next, "Death", genome_ID, agent))
                        
                        self.X = classification_S_M_PC(transitionPlayer=agent,
                                                       X_matrix=self.X,
                                                       mature_matrix=self.mature_matrix)
                    elif evt_type == "Death":
                        # Quita 1 clon de mature
                        self.mature_matrix[genome_ID, agent] -= 1
                        
                        self.X = classification_S_M_PC(transitionPlayer=agent,
                                                       X_matrix=self.X,
                                                       mature_matrix=self.mature_matrix)

                    #  Podar haplotipos extintos tras cada muerte de parásito #
                    pop_matrix = update_matrices(mature_matrix = self.mature_matrix,
                                                 immature_matrix = self.immature_matrix,
                                                 parasitic_populations = self.parasitic_populations)

                    self.parasitic_populations = pop_matrix[0]
                    self.mature_matrix = pop_matrix[1]
                    self.immature_matrix = pop_matrix[2]

            #####################
            ##  Run the model ##
            #####################
            
            self.calculate_forces()
            self.next_time_event()
            self.variate_population()
 
            
            # 2) Guardar stats por cada unidad de tiempo superada
            while self.actual_time >= time_step and time_step <= tmax:
                ratio = (self.generation_events / self.total_events) if self.total_events > 0 else 0
                self.save_information(time_step, ratio, len(self.parasitic_populations))
                time_step += 1