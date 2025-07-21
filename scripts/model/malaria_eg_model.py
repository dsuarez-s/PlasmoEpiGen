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
from scripts.transitions.humans_events    import human_to_mosquito, classification_S_M_PC

# Recombination y limpieza de haplotipos #
from scripts.genetics.recombination import recombination, update_matrices

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
        self.t = 0
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
        self.genomes_matrix = init_genomes[1]
        self.pre_genomes_matrix = init_genomes[2]  
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
                                     parasitic_populations = self.parasitic_populations,
                                     pre_genomes_matrix = self.pre_genomes_matrix,
                                     genomes_matrix = self.genomes_matrix,
                                     epi_params = self.epi)
            for g in inoc:
                if self.pre_genomes_matrix[g, self.transitionPlayer] == 0:
                    self.pre_genomes_matrix[g, self.transitionPlayer] = self.alpha_H

        elif self.transitionType == "lambda_mosquitoes":
            inoc = human_to_mosquito(X = self.X,
                                     parasitic_populations = self.parasitic_populations,
                                     pre_genomes_matrix=self.pre_genomes_matrix,
                                     genomes_matrix=self.genomes_matrix,
                                     epi_params=self.epi)
            
            result = recombination(inoculated_genomes=inoc, 
                                   parasitic_populations=self.parasitic_populations,
                                   pre_genomes_matrix=self.pre_genomes_matrix,
                                   genomes_matrix=self.genomes_matrix,
                                   total_events=self.total_events,
                                   generation_events=self.generation_events,
                                   distribution=self.config["distribution"])
            
            self.parasitic_populations = result[0]
            self.pre_genomes_matrix = result[1]
            self.genomes_matrix = result[2]
            self.total_events = result[3]
            self.generation_events = result[4]
            selected = result[5]
            
            for g in selected:
                if self.pre_genomes_matrix[g, self.transitionPlayer] == 0:
                    self.pre_genomes_matrix[g, self.transitionPlayer] = self.alpha_M

        else:
            # Mosquito death/reset to susceptible #
            matrices = func_toMS(transitionPlayer = self.transitionPlayer,
                                 X_matrix = self.X,
                                 pre_genomes_matrix = self.pre_genomes_matrix,
                                 genomes_matrix = self.genomes_matrix)
            
            self.X = matrices[0]
            self.genomes_matrix = matrices[1]
            self.pre_genomes_matrix = matrices[2]
    
            pop_matrix = update_matrices(genomes_matrix = self.genomes_matrix,
                                         pre_genomes_matrix = self.pre_genomes_matrix,
                                         parasitic_populations = self.parasitic_populations)

            self.parasitic_populations = pop_matrix[0]
            self.genomes_matrix = pop_matrix[1]
            self.pre_genomes_matrix = pop_matrix[2]
    
    
    # -------------------------------- #
    # Part 6: Saving Simulation Output #
    # -------------------------------- #
    def save_information(self, time_step, ratio_reco, num_haplotypes):
        # Append summary of the current state to a file #
        folder = self.config["name_folder"]
        if not os.path.exists(folder):
            os.mkdir(folder)
            
        fname   = f"BR_{self.epi["beta_hv"]}_IGD_{self.config["size_pool"]}_{self.config["iteration"]}.txt"
        path = os.path.join(folder, fname)
        
        # Conteo de Estados #
        nums = [ (self.X == s).sum() for s in (self.HS, self.HM, self.HPC, self.MS, self.MC, self.MPC)       ]
        
        # Fila a Escribir #
        row = (f"{time_step};" +  ";".join(str(n) for n in nums) + ";" + f"{ratio_reco};{num_haplotypes}")
        with open(path, "a") as f:
            f.write(row)
            
    # ---------------------------- #
    # Part 7: Main Simulation Loop #
    # ---------------------------- #
    def run(self, tmax):
        # Run the full simulation up to tmax #
        time_step = 1
        self.calculate_forces()
        
        #self.save_information(0, 0, self.genomes_matrix.shape[0])

        while self.t < tmax:
            self.calculate_forces()
            self.next_time_event()
            self.variate_population()
            self.t += self.tau

            # Decrease timers #
            self.genomes_matrix -= (self.genomes_matrix > 0) * self.tau
            self.pre_genomes_matrix -= (self.pre_genomes_matrix > 0) * self.tau

            # Handle pre-genome to genome transitions #
            nz_pre = sparse.find(self.pre_genomes_matrix)
            nz_gen = sparse.find(self.genomes_matrix)
            new_inf = np.where(nz_pre[2] < 0)[0]
            new_rec = np.where(nz_gen[2] < 0)[0]
            agents = set()

            for idx in new_inf:
                g, ag = nz_pre[0][idx], nz_pre[1][idx]
                agents.add(ag)
                self.pre_genomes_matrix[g, ag] = 0
                self.pre_genomes_matrix.eliminate_zeros()
                timer = self.epi["gamma"] if ag < self.MS else self.epi["xi"]
                self.genomes_matrix[g, ag] = timer

            for idx in new_rec:
                g, ag = nz_gen[0][idx], nz_gen[1][idx]
                agents.add(ag)
                self.genomes_matrix[g, ag] = 0
                self.genomes_matrix.eliminate_zeros()


            
            up_pop_mat = update_matrices(genomes_matrix=self.genomes_matrix,
                                         pre_genomes_matrix=self.pre_genomes_matrix,
                                         parasitic_populations=self.parasitic_populations)
            
            self.parasitic_populations = up_pop_mat[0]
            self.genomes_matrix = up_pop_mat[1]
            self.pre_genomes_matrix = up_pop_mat[2] 
            
            humans_pos = set(np.where(self.X < self.MS)[0])
            for ag in agents:
                classification_S_M_PC(ag, self.genomes_matrix)

            if self.t > time_step:
                ratio = (self.generation_events / self.total_events if self.total_events > 0 else 0)
                self.save_information(time_step, ratio, self.genomes_matrix.shape[0])
                time_step += 1
