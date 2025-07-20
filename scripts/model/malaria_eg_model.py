# Main Malaria Epidemiological and Genetic Model #
import numpy as np
import random
import os
from scipy import sparse
from scripts.genetics.recombination.genome_initialization import initialize_genomes
from scripts.transitions.calculate_forces import compute_propensities


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
    
class MalariaEGModel:
    def __init__(self, epi_parameters, pop_parameters,
                 name_folder, size_pool, iteration, distribution, genomes,
                 clone_distribution_human, clone_distribution_mosquito):
        
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
        self.pop = initialize_population_counts(pop_parameters)

        # Agent state vector #
        self.X = np.array([self.HS] * HS_init + [self.HM] * HM_init + [self.HPC] * HPC_init + 
                          [self.MS] * MS_init + [self.MC] * MC_init + [self.MPC] * MPC_init)

        np.random.shuffle(self.X)

        # Event list and counters #
        self.events = ["lambda_humans", "lambda_mosquitoes", "toMS"]
        self.generation_events = 0
        self.total_events = 0
        
    def initalize_genomes(self):
        
        init_genomes = initialize_genomes(X = self.X, gamma = self.gamma , xi = self.xi,
                                          genomes_dictionary = genomes,
                                          HM_code = self.HM, HPC_code = self.HPC,
                                          MC_code = self.MC, MPC_code = self.MPC,
                                          clone_distribution_human = clone_distribution_human,
                                          clone_distribution_mosquito = clone_distribution_mosquito)
        
        self.parasitic_populations = init_genomes[0]
        self.genomes_matrix = init_genomes[1]
        self.pre_genomes_matrix = init_genomes[2]
    
    def calculate_forces(self):
        self.propensities = compute_propensities(X = self.X, Hum_Pop = self.Hum_Pop, Mos_Pop = self.Mos_Pop,
                                                 sigma_v = self.sigma_v, sigma_h = self.sigma_h,
                                                 beta_hv = self.beta_hv, beta_vh = self.beta_vh,
                                                 delta = self.delta)
    def next_time_event(self):
        # Determine next event and time increment #
        total = self.propensities.sum()
        cum = np.cumsum(self.propensities)
        r = np.random.rand()
        self.tau = np.random.exponential(1 / total)
        idx = np.searchsorted(cum, r * total)
        self.transitionPlayer = idx % len(self.X)
        self.transitionType = self.events[idx // len(self.X)]

    def variate_population(self):
        # Apply state transitions based on event type #
        if self.transitionType == "lambda_humans":
            inoc = self.mosquito_to_human()
            for g in inoc:
                if self.pre_genomes_matrix[g, self.transitionPlayer] == 0:
                    self.pre_genomes_matrix[g, self.transitionPlayer] = self.alpha_H

        elif self.transitionType == "lambda_mosquitoes":
            inoc = self.human_to_mosquito()
            result = recombination(inoculated_genomes=inoc, 
                                   parasitic_populations=self.parasitic_populations,
                                   pre_genomes_matrix=self.pre_genomes_matrix,
                                   genomes_matrix=self.genomes_matrix,
                                   total_events=self.total_events,
                                   generation_events=self.generation_events,
                                   distribution=self.distribution)
            
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
            matrices = func_toMS(transitionPlayer=self.transitionPlayer,
                                 X_matrix=self.X,
                                 pre_genomes_matrix=self.pre_genomes_matrix,
                                 genomes_matrix=self.genomes_matrix)
            
            self.X = matrices[0]
            self.genomes_matrix = matrices[1]
            self.pre_genomes_matrix = matrices[2]
    
            pop_matrix = update_matrices(genomes_matrix=self.genomes_matrix,
                                         pre_genomes_matrix=self.pre_genomes_matrix,
                                         parasitic_populations=self.parasitic_populations)

            self.parasitic_populations = pop_matrix[0]
            self.genomes_matrix = pop_matrix[1]
            self.pre_genomes_matrix = pop_matrix[2]

    def save_information(self, time_step, ratio_reco, num_haplotypes):
        # Append summary of the current state to a file #
        folder = self.name_folder
        if not os.path.exists(folder):
            os.mkdir(folder)
        fname = f"BR_{self.a}_IGD_{self.size_pool}_{self.iteration}.txt"
        path = os.path.join(folder, fname)
        nums = [ (self.X == s).sum() for s in (self.HS, self.HM, self.HPC, self.MS, self.MC, self.MPC)       ]
        row = (f"{time_step};" +  ";".join(str(n) for n in nums) + ";" + f"{ratio_reco};{num_haplotypes}")
        with open(path, "a") as f:
            f.write(row)

    def run(self, tmax, genomes):
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
                timer = self.gamma if ag < self.MS else self.xi
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
