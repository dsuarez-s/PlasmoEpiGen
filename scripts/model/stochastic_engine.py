import numpy as np
import heapq
from scripts.transitions.mosquitoes_events import func_toMS, mosquito_to_human
from scripts.transitions.humans_events import human_to_mosquito, func_toHS
from scripts.helpers.state_inspectors import classification_S_M_PC
from scripts.genetics.recombination import recombination

"""
sigma_h:   Rate of bites per mosquito 
gamma:     Recovery time of a human from infection
delta:     Lifespan of mosquitoes
alpha_H:   Maturation time of gametocytes in humans
alpha_M:   Maturation time of sporozoites in mosquitoes
sigma_v:   Rate of gonotrophic cycle (mosquito feeding cycle)
beta_hv:   Probability of transmission from mosquito to human
beta_vh:   Probability of transmission from human to mosquito
"""

# --------------------------------------------------------------------------------------------- #

def compute_propensities(self):    
    # Paso 1: Calculamos las fuerzas de infección de humanos y vectores #
    # Humanos #
    inf_mos = (self.X == self.MC) | (self.X == self.MPC)
    prop_bites_h_num = (self.epi["sigma_v"] * self.num_mos * self.epi["sigma_h"])
    prop_bites_h_den =(self.epi["sigma_v"] * self.num_mos + self.epi["sigma_h"] * self.num_hum)
    prop_bites_h = prop_bites_h_num/prop_bites_h_den
    lambda_h = prop_bites_h * self.epi["beta_hv"] * (inf_mos.sum() / self.num_mos)
    # Mosquitos #
    inf_hum = (self.X == self.HM) | (self.X == self.HPC)
    prop_bites_m_num = (self.epi["sigma_v"] * self.epi["sigma_h"] * self.num_hum)
    prop_bites_m_den = (self.epi["sigma_v"] * self.num_mos + self.epi["sigma_h"] * self.num_hum)
    prop_bites_m = prop_bites_m_num/prop_bites_m_den
    lambda_v = prop_bites_m * self.epi["beta_vh"] * (inf_hum.sum() / self.num_hum)
    
    # Paso 2: A cada agente con estados específicos se les asigna la propensity correspondiente #
    # Humanos #
    all_hum = (self.X == self.HS) | (self.X == self.HM) | (self.X == self.HPC)
    prop_inf_h = lambda_h * all_hum # Infección Humano #
    prop_clearance_hum = (1 / self.epi["gamma"]) * inf_hum # Recuperación Humano #
    # Mosquitos #
    all_mos = (self.X == self.MS) | (self.X == self.MC) | (self.X == self.MPC)
    prop_inf_v = lambda_v * all_mos # Infección Mosquito #
    prop_death_mos = (1 / self.epi["delta"]) * all_mos # Muerte Mosquito #
    
    # Paso 3: Se guarda un vector con todos los valores obtenidos para los eventos en orden #
    self.propensities = np.hstack([prop_inf_h, prop_inf_v, prop_death_mos, prop_clearance_hum])
    return(self.propensities)

# --------------------------------------------------------------------------------------------- #
# Calculamos el proximo tiempo, evento y agente al que le ocurrirá #
def next_time_event(self):
    total = self.propensities.sum()
    if total <= 0:
        self.tau = 1
        self.transitionPlayer = None
        self.transitionType = None
        return (self.tau, self.transitionPlayer, self.transitionType)
    else:
        cum = np.cumsum(self.propensities)
        r = np.random.rand()
        self.tau = np.random.exponential(1 / total)
        idx = np.searchsorted(cum, r * total)
        self.transitionPlayer = idx % len(self.X)
        self.transitionType = self.events[idx // len(self.X)]
        return (self.tau, self.transitionPlayer, self.transitionType)

# --------------------------------------------------------------------------------------------- #

def variate_population(self):
    
    # Check 1: El primer evento es la infección de un humano dado por un mosquito #
    if self.transitionType == "lambda_humans":
        # Paso 1: Se escoge los genomas que serán inoculados #
        inoc = mosquito_to_human(X = self.X, 
                                 mature_matrix = self.mature_matrix,
                                 MC_code = self.MC, MPC_code = self.MPC)
        
        # Paso 2: Se recorren los genomas inoculados y se agregan a la matriz de parásitos inmaduros #
        for g in inoc:
            self.immature_matrix[g, self.transitionPlayer] += 1
            genome_ID = g
            t_event = self.epi["alpha_H"] + self.actual_time 
            type_event = "Gametocytes Maturation"
            agent = self.transitionPlayer
            heapq.heappush(self.event_queue, (t_event, type_event, genome_ID, agent))

    # Check 2: El segundo evento es la infección de un mosquito al picar un humano #
    elif self.transitionType == "lambda_mosquitoes":
        # Paso 1: Se escoge los genomas que serán inoculados #
        inoc = human_to_mosquito(X = self.X,
                                 mature_matrix=self.mature_matrix,
                                 HM_code=self.HM, HPC_code =self.HPC)
        
        # Paso 2: Se aumenta el contador de eventos de infección que han sufrido los mosquitos #
        self.total_events_infect += 1
        prev_generation_events = self.generation_events

        # Paso 3: Se lleva a cabo el proceso de recombinación si se tienen las condiciones adecuadas #
        result = recombination(inoculated_genomes=inoc, 
                               parasitic_populations=self.parasitic_populations,
                               immature_matrix=self.immature_matrix,
                               mature_matrix=self.mature_matrix,
                               generation_events=self.generation_events,
                               dist_loci=self.config["distribution"])

        self.parasitic_populations = result[0]
        self.immature_matrix = result[1]
        self.mature_matrix = result[2]
        self.generation_events = result[3]
        selected = result[4]
        
        # Check 2.1 Se revisa si hubo recombinación en este evento de infección #     
        if self.generation_events > prev_generation_events:
            self.infect_with_reco += 1
        
        # Paso 3: Se recorren los genomas inoculados y se agregan a la matriz de parásitos inmaduros #
        for g in selected:
            self.immature_matrix[g, self.transitionPlayer] += 1
            genome_ID = g
            t_event = self.epi["alpha_M"] + self.actual_time 
            type_event = "Sporozoites Maturation"
            agent = self.transitionPlayer
            heapq.heappush(self.event_queue, (t_event, type_event, genome_ID, agent))

    # Check 3: El tercer evento es la recuperación de un humano de la infección #
    elif self.transitionType == "human_clearance":
        # Paso 1: Limpiar las matrices de mature e immature y actualizar el estado en X #
        matrices_event_queue = func_toHS(transition_Player = self.transitionPlayer,
                                         X_matrix = self.X,
                                         immature_matrix = self.immature_matrix,
                                         mature_matrix = self.mature_matrix,
                                         HS_code = self.HS,
                                         event_queue = self.event_queue)
        
        self.X,self.mature_matrix, self.immature_matrix,self.event_queue  = matrices_event_queue
    
    # Check 4: El cuarto evento es la muerte de un mosquito #
    else:
        # Paso 1: Limpiar las matrices de mature e immature y actualizar el estado en X #
        matrices_event_queue = func_toMS(transition_Player = self.transitionPlayer,X_matrix = self.X,
                                        immature_matrix = self.immature_matrix,
                                        mature_matrix = self.mature_matrix,
                                        MS_code = self.MS,
                                        event_queue = self.event_queue)
        
        self.X,self.mature_matrix, self.immature_matrix,self.event_queue  = matrices_event_queue