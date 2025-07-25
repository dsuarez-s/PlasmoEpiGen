# Fast run script for MalariaEGModel #
from scripts.model.malaria_eg_model import MalariaEGModel
import numpy as np

##############################
# Parameters and References ##
##############################

sigma_h = 40    # Number of bites per mosquito  #
gamma = 50     # Recovery time of a human from infection [5] #
delta = 90.4    # Lifespan of mosquitoes [4] #
alpha_H = 11    # Maturation time of gametocytes in humans [2] #
alpha_M = 14    # Maturation time of sporozoites in mosquitoes [1] #
sigma_v = 3.1   # Rate of gonotrophic cycle (mosquito feeding cycle) [3] #
beta_hv = 0.48  # Probability of transmission from mosquito to human [5] #
beta_vh = 0.022 # Probability of transmission from human to mosquito [5] #
xi = 55         # Lifespan of parasites in mosquito salivary glands [4] #
xi = 25         # Lifespan of parasites in mosquito salivary glands [4] #

# References #

# [1] Dong, Shengzhang, et al. Trends in Parasitology (2021).
# [2] Sauerwein, Meta Roestenberg, Moorthy. Nature Rev Immunol (2011).
# [3] Rúa et al. Memórias Do Instituto Oswaldo Cruz (2005).
# [4] Mayne. Public Health Reports (1922).
# [5] Chitnis, Hyman, Cushing. Bull Math Biol (2008).

#################
# Run the Model #
#################

initial_genomes = { 0: "AAAAAA", 1: "BBBBBB" }
dist_humans = {0: 0.25, 1:0.5 , 2:0.25} 
dist_mosquitoes = {0: 0.4, 1:0.2 , 2:0.4} 
epidemiological_parameters = [sigma_h, gamma, delta, alpha_H, alpha_M, sigma_v, beta_hv, beta_vh, xi]    
population_parameters = {"Mos": 10 , "Hum": 10}

# ------------------------------------------------------------------ #
model = MalariaEGModel(epi_parameters = epidemiological_parameters,
                       pop_parameters = population_parameters,
                       name_folder="test",
                       iteration="proof",
                       distribution=[1/6] * 6,
                       genomes = initial_genomes,
                       clone_distribution_human = dist_humans,
                       clone_distribution_mosquito = dist_mosquitoes)
# ------------------------------------------------------------------ #
model.run(tmax=60)
# ------------------------------------------------------------------ #      