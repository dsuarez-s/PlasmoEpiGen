# Main Malaria Epidemiological and Genetic Model #
import numpy as np
import random
import os
from scipy import sparse
from scripts.transitions.mosquitoes_events import *
from scripts.transitions.humans_events import *
from scripts.recombination.recombination import *

class MalariaEGModel:
    def __init__(self, epi_parameters, pop_parameters,
                 name_folder, size_pool, iteration, distribution):
        # Initialize model parameters #
        self.name_folder = name_folder
        self.size_pool = size_pool
        self.iteration = iteration
        self.distribution = distribution

        # Parasite cycle parameters #
        self.alpha_H = epi_parameters[0]
        self.alpha_M = epi_parameters[1]
        self.sigma = epi_parameters[2]

        # Epidemiological parameters #
        self.a = epi_parameters[3]
        self.b = epi_parameters[4]
        self.c = epi_parameters[5]
        self.xi = epi_parameters[6]
        self.mu = epi_parameters[7]
        self.gamma = epi_parameters[8]

        # Time and state counters #
        self.t = 0
        self.HS, self.HM, self.HPC = 0, 1, 2
        self.MS, self.MC, self.MPC = 3, 4, 5

        # Population sizes #
        self.Hum_Pop = int(pop_parameters[0])
        self.Mos_Pop = int(pop_parameters[1])

        # Agent state vector #
        self.X = np.array(
            [self.HM] * int(pop_parameters[2]) +
            [self.HS] * (self.Hum_Pop - pop_parameters[2]) +
            [self.MC] * int(pop_parameters[3]) +
            [self.MS] * (self.Mos_Pop - pop_parameters[3])
        )
        np.random.shuffle(self.X)

        # Event list and counters #
        self.events = ["lambda_humans", "lambda_mosquitoes", "toMS"]
        self.generation_events = 0
        self.total_events = 0

    def initial_genomes(self, genomes):
        # Assign initial parasite genomes to agents #
        current_HM = np.where(self.X == self.HM)[0]
        current_MC = np.where(self.X == self.MC)[0]
        tot_init_inf = len(current_HM) + len(current_MC)
        tot_list_inf = np.concatenate([current_HM, current_MC])

        # Prepare genome pool #
        if len(genomes) < tot_init_inf:
            reps = tot_init_inf // len(genomes) + 1
            gen_f = list(genomes.keys()) * reps
        else:
            gen_f = list(genomes.keys())

        h_selected = random.sample(gen_f, tot_init_inf)

        # Initialize matrices #
        unique = set(h_selected)
        self.parasitic_populations = np.array([])
        size = len(unique)
        N = len(self.X)
        self.genomes_matrix = sparse.csr_matrix((size, N), dtype=int)
        self.pre_genomes_matrix = sparse.csr_matrix((size, N), dtype=int)

        mapping = {}
        for idx, val in enumerate(unique):
            mapping[val] = idx
            seq = genomes[val]
            self.parasitic_populations = np.append(
                self.parasitic_populations,
                "".join(seq) if not isinstance(seq, str) else seq
            )

        for i, h in enumerate(h_selected):
            pos = mapping[h]
            agent = tot_list_inf[i]
            timer = self.gamma if agent in current_HM else self.xi
            self.genomes_matrix[pos, agent] = timer

    def calculate_forces(self):
        # Compute propensities for infection and death events #
        sigma_v = 1 / self.sigma
        active_mos = (self.X == self.MC) | (self.X == self.MPC)
        inf_hum = (self.X == self.HM) | (self.X == self.HPC)
        sus_hum = (self.X == self.HS) | inf_hum

        prop_bites_h = (sigma_v * self.Mos_Pop * self.a) / (
            sigma_v * self.Mos_Pop + self.a * self.Hum_Pop
        )
        lambda_humans = prop_bites_h * self.b *             (active_mos.sum() / self.Mos_Pop) * sus_hum

        sus_mos = (self.X == self.MS) | (self.X == self.MC) | (self.X == self.MPC)
        prop_bites_m = (sigma_v * self.a * self.Hum_Pop) / (
            sigma_v * self.Mos_Pop + self.a * self.Hum_Pop
        )
        lambda_mosquitoes = prop_bites_m * self.c *             (inf_hum.sum() / self.Hum_Pop) * sus_mos

        toMS = (1 / self.mu) * active_mos

        self.propensities = np.hstack([lambda_humans,
                                       lambda_mosquitoes,
                                       toMS])

    def next_time_event(self):
        # Determine next event and time increment #
        total = self.propensities.sum()
        cum = np.cumsum(self.propensities)
        r = np.random.rand()
        self.tau = np.random.exponential(1 / total)
        idx = np.searchsorted(cum, r * total)
        self.transitionPlayer = idx % len(self.X)
        self.transitionType = self.events[idx // len(self.X)]

    def mosquito_to_human(self):
        # Select genomes from mosquito to infect human #
        mos = random.choice(np.where(
            (self.X == self.MC) | (self.X == self.MPC))[0])
        gens = self.genomes_matrix[:, mos].tocoo().row
        if len(gens) <= 1:
            return list(gens)
        return random.sample(list(gens), random.randint(1, len(gens)))

    def human_to_mosquito(self):
        # Select genomes from human to infect mosquito #
        hum = random.choice(np.where(
            (self.X == self.HM) | (self.X == self.HPC))[0])
        gens = self.genomes_matrix[:, hum].tocoo().row
        if len(gens) <= 1:
            return list(gens)
        return random.sample(list(gens), random.randint(1, len(gens)))

    def variate_population(self):
        # Apply state transitions based on event type #
        if self.transitionType == "lambda_humans":
            inoc = self.mosquito_to_human()
            for g in inoc:
                if self.pre_genomes_matrix[g, self.transitionPlayer] == 0:
                    self.pre_genomes_matrix[g, self.transitionPlayer] = self.alpha_H

        elif self.transitionType == "lambda_mosquitoes":
            inoc = self.human_to_mosquito()
            result = recombination(
                inoculated_genomes=inoc,
                parasitic_populations=self.parasitic_populations,
                pre_genomes_matrix=self.pre_genomes_matrix,
                genomes_matrix=self.genomes_matrix,
                total_events=self.total_events,
                generation_events=self.generation_events,
                distribution=self.distribution
            )
            (self.parasitic_populations,
             self.pre_genomes_matrix,
             self.genomes_matrix,
             self.total_events,
             self.generation_events,
             selected) = result
            for g in selected:
                if self.pre_genomes_matrix[g, self.transitionPlayer] == 0:
                    self.pre_genomes_matrix[g, self.transitionPlayer] = self.alpha_M

        else:
            # Mosquito death/reset to susceptible #
            self.X, self.genomes_matrix, self.pre_genomes_matrix = func_toMS(
                transitionPlayer=self.transitionPlayer,
                X_matrix=self.X,
                pre_genomes_matrix=self.pre_genomes_matrix,
                genomes_matrix=self.genomes_matrix
            )
            (self.parasitic_populations,
             self.genomes_matrix,
             self.pre_genomes_matrix) = update_matrices(
                genomes_matrix=self.genomes_matrix,
                pre_genomes_matrix=self.pre_genomes_matrix,
                parasitic_populations=self.parasitic_populations
            )

    def save_information(self, time_step, ratio_reco, num_haplotypes):
        # Append summary of the current state to a file #
        folder = self.name_folder
        if not os.path.exists(folder):
            os.mkdir(folder)
        fname = f"BR_{self.a}_IGD_{self.size_pool}_{self.iteration}.txt"
        path = os.path.join(folder, fname)
        nums = [
            (self.X == s).sum()
            for s in (self.HS, self.HM, self.HPC,
                      self.MS, self.MC, self.MPC)
        ]
        row = (f"{time_step};" +
               ";".join(str(n) for n in nums) + ";" +
               f"{ratio_reco};{num_haplotypes}
")
        with open(path, "a") as f:
            f.write(row)

    def run(self, tmax, genomes):
        # Run the full simulation up to tmax #
        time_step = 1
        self.initial_genomes(genomes)
        self.calculate_forces()
        self.save_information(0, 0,
                              self.genomes_matrix.shape[0])

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

            (self.parasitic_populations,
             self.genomes_matrix,
             self.pre_genomes_matrix) = update_matrices(
                genomes_matrix=self.genomes_matrix,
                pre_genomes_matrix=self.pre_genomes_matrix,
                parasitic_populations=self.parasitic_populations
            )

            humans_pos = set(np.where(self.X < self.MS)[0])
            for ag in agents:
                classification_S_M_PC(ag, self.genomes_matrix)

            if self.t > time_step:
                ratio = (self.generation_events / self.total_events
                         if self.total_events > 0 else 0)
                self.save_information(time_step, ratio,
                                      self.genomes_matrix.shape[0])
                time_step += 1
