# -------------------------------- #
# Part 1: Imports and Dependencies #
# -------------------------------- #
import numpy as np
import random
import os
from scipy import sparse
import heapq
from collections import defaultdict
import pickle


# Initialization # 
from model_init import init_model_state

# Stochastic Dynamics #
from model.stochastic_engine import compute_propensities, next_time_event, variate_population

# Helpers para populación y genomas #
from scripts.genetics.genome_initialization import initialize_genomes

# Cálculo de tasas #
from scripts.transitions.calculate_forces import compute_propensities

# Funciones de transición de estado #
from scripts.transitions.mosquitoes_events import func_toMS, mosquito_to_human
from scripts.transitions.humans_events import human_to_mosquito, func_toHS
from scripts.transitions.event_queue_schedule import event_queue_execution

# Recombination y limpieza de haplotipos #
from scripts.genetics.recombination import recombination
from scripts.helpers.state_inspectors import update_matrices, classification_S_M_PC

# Metrics measured along the process #
from scripts.observables.identity_by_descent import (precompute_ibd_table, measure_ibd_relative_to_founders as measure_ibd)
from scripts.observables.nucleotide_diversity import (measure_nucleotide_diversity as measure_pi)
from scripts.observables.shannon_index import (measure_shannon_population as measure_shannon)
from scripts.observables.multiplicity_of_infection import measure_moi

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
                 name_folder, iteration, distribution, genomes,
                 clone_distribution_human, clone_distribution_mosquito):
        
        # Model Initialization (__init__) #
        init_model_state(self,epi_parameters, pop_parameters,name_folder,
                         iteration, distribution, genomes, clone_distribution_human,
                         clone_distribution_mosquito, heapq=heapq)
          
    # Computing Propensities # 
    def compute_propensities(self):
        return compute_propensities(self)

    # Selecting Next Event #
    def next_time_event(self):
        return next_time_event(self)

    # Applying State Transitions #
    def variate_population(self, event_index):
        return variate_population(self, event_index)    
    
    # Saving Simulation Output #
    def save_information(self, folder=None):
        return save_information(self, folder)
    
    # Main Simulation Loop #
    def run(self, tmax):
        return run(self, tmax)             