# Main Malaria Epidemiological and Genetic Model #
import numpy as np
import random
import os
from scipy import sparse
from scripts.transitions.mosquitoes_events import *
from scripts.transitions.humans_events import *
from scripts.recombination.recombination import *

class MalariaEGModel:
    def __init__(self, epi_parameters, pop_parameters, name_folder, size_pool, iteration, distribution):
        
        # Initialize model parameters #
        self.name_folder = name_folder
        self.size_pool = size_pool
        self.iteration = iteration
        self.distribution = distribution

        # Epidemiological parameters #
        self.sigma_h = epi_parameters[0] # Number of bites #
        self.gamma = epi_parameters[1] # Recovery time of a human from an infection #
        self.delta = epi_parameters[2] # Lifespam of mosquitoes #

        # Parasite and Mosquito cycle parameters #
        self.alpha_H = epi_parameters[3] # Human Infection: Maturation of Gametocytes #
        self.alpha_M = epi_parameters[4] # Mosquito Infection: Maturation of Sporozoites #
        self.sigma_v = epi_parameters[5] # Gonotrophic Cycle #
        self.beta_hv = epi_parameters[6] # Prob. Human Infection from Vector # 
        self.beta_vh = epi_parameters[7] # Prob. Vector Infection from Human #
        self.xi = epi_parameters[8] # Lifespam of parasites in salivary glands #      

        # Time and state counters #
        self.t = 0
        self.HS, self.HM, self.HPC = 0, 1, 2
        self.MS, self.MC, self.MPC = 3, 4, 5

        # Population sizes #
        self.Hum_Pop = int(pop_parameters[0]) # Number of Humans #
        self.Mos_Pop = int(pop_parameters[1]) # Number of Mosquitoes #
        
        HM_init = int(self.Hum_Pop*pop_parameters[2])  # Initial number of Monoclonal infected humans #
        HPC_init = int(self.Hum_Pop*pop_parameters[3]) # Initial number of Policlonal infected humans #
        HS_init = self.Hum_Pop - HM_init - HPC_init    # Initial number of Suceptible humans #
        
        MC_init = int(self.Mos_Pop*pop_parameters[4])  # Initial number of Monoclonal infected mosquitoes #
        MPC_init = int(self.Mos_Pop*pop_parameters[5]) # Initial number of Policlonal infected mosquitoes #
        MS_init = self.Mos_Pop - MC_init - MPC_init    # Initial number of Suceptible mosquitoes #
        
        assert HS_init >= 0 and MS_init >= 0, "Initial population fractions exceed total population."

        # Agent state vector #
        self.X = np.array([self.HS] * HS_init + [self.HM] * HM_init + [self.HPC] * HPC_init + 
                          [self.MS] * MS_init + [self.MC] * MC_init + [self.MPC] * MPC_init)

        np.random.shuffle(self.X)

        # Event list and counters #
        self.events = ["lambda_humans", "lambda_mosquitoes", "toMS"]
        self.generation_events = 0
        self.total_events = 0
    
    #############################
    ### Continue Here Working ### 
    #############################
    
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
        self.initial_genomes(genomes)
        self.calculate_forces()
        self.save_information(0, 0, self.genomes_matrix.shape[0])

        while self.t < tmax:
            print(self.parasitic_populations)
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
