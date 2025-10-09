import numpy as np

def compute_propensities(self):
    # División para cada uno de los estados ya sea infección o susceptibilidad #
    inf_mos = (self.X == self.MC) | (self.X == self.MPC)
    inf_hum = (self.X == self.HM) | (self.X == self.HPC)
    sus_hum = (self.X == self.HS) | (self.X == self.HM) | (self.X == self.HPC)
    sus_mos = (self.X == self.MS) | (self.X == self.MC) | (self.X == self.MPC)
    
    # Calculamos las fuerzas de infección de humanos y vectores #
    prop_bites_h = (self.epi["sigma_v"] * self.num_mos * self.epi["sigma_h"]) / (
        self.epi["sigma_v"] * self.num_mos + self.epi["sigma_h"] * self.num_hum
    )
    prop_bites_m = (self.epi["sigma_v"] * self.epi["sigma_h"] * self.num_hum) / (
        self.epi["sigma_v"] * self.num_mos + self.epi["sigma_h"] * self.num_hum
    )

    lambda_h = prop_bites_h * self.epi["beta_hv"] * (inf_mos.sum() / self.num_mos)
    lambda_v = prop_bites_m * self.epi["beta_vh"] * (inf_hum.sum() / self.num_hum)
    
    # Calculamos la propensity para cada uno de los eventos descritos #
    prop_inf_h = lambda_h * sus_hum
    prop_inf_v = lambda_v * sus_mos
    prop_death_mos = (1 / self.epi["delta"]) * inf_mos
    prop_clearance_hum = (1 / self.epi["gamma"]) * inf_hum
    
    # Guardar en el objeto del modelo
    self.propensities = np.hstack([prop_inf_h, prop_inf_v, prop_death_mos, prop_clearance_hum])

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
        inoc = mosquito_to_human(X = self.X, 
                                 mature_matrix = self.mature_matrix,
                                 MC_code = self.MC, MPC_code = self.MPC)

        for g in inoc:
            if self.immature_matrix[g, self.transitionPlayer] == 0:
                self.immature_matrix[g, self.transitionPlayer] += 1

                # Schedule a death event for each assigned haplotype
                genome_ID = g
                t_event = self.epi["alpha_H"] + self.actual_time 
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
                t_event = self.epi["alpha_M"] + self.actual_time 
                type_event = "Sporozoites Maturation"
                agent = self.transitionPlayer
                heapq.heappush(self.event_queue, (t_event, type_event, genome_ID, agent))

        self.X = classification_S_M_PC(transitionPlayer=self.transitionPlayer,
                                       X_matrix=self.X,
                                       mature_matrix=self.mature_matrix)

    elif self.transitionType == "human_clearance":

        matrices = func_toHS(transition_Player = self.transitionPlayer,
                             X_matrix = self.X,
                             immature_matrix = self.immature_matrix,
                             mature_matrix = self.mature_matrix,
                             HS_code = self.HS)

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

    else:
        # Mosquito death/reset to susceptible #
        matrices = func_toMS(transition_Player = self.transitionPlayer,
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