# Fast run script for MalariaEGModel #
from scripts.model.malaria_eg_model import MalariaEGModel
import numpy as np
import sys

iter_number = sys.argv[1]
init_gen_div = int(sys.argv[2])
num_bites = float(sys.argv[3])
num_hum = int(sys.argv[4])
mos_x_hum = int(sys.argv[5])

##############################
# Parameters and References ##
##############################

sigma_h = float(num_bites)    # Number of bites per mosquito  #
gamma = 289     # Recovery time of a human from infection [5] #
delta = 30    # Lifespan of mosquitoes [4] #
alpha_H = 11    # Maturation time of gametocytes in humans [2] #*
alpha_M = 14    # Maturation time of sporozoites in mosquitoes [1] #*
sigma_v = 1/3.1   # Rate of gonotrophic cycle (mosquito feeding cycle) [3] #
beta_hv = 0.2  # Probability of transmission from mosquito to human [5] #
beta_vh = 0.07 # Probability of transmission from human to mosquito [5] #

# References #
# [1] Dong, Shengzhang, et al. Trends in Parasitology (2021).
# [2] Sauerwein, Meta Roestenberg, Moorthy. Nature Rev Immunol (2011).
# [3] Rúa et al. Memórias Do Instituto Oswaldo Cruz (2005).
# [4] Mayne. Public Health Reports (1922).
# [5] Chitnis, Hyman, Cushing. Bull Math Biol (2008).

#################
# Run the Model #
#################
if (init_gen_div == 10):
    initial_genomes = {0: "A"*75, 1: "A"*60 + "B"*15}
    
elif (init_gen_div == 30):
    initial_genomes = {0: "A"*75, 1: "A"*30 + "B"*45}

elif (init_gen_div == 50):
    initial_genomes = {0: "A"*75, 1: "B"*75}

elif (init_gen_div == 75):
    initial_genomes = {0: "A"*75, 1: "B"*75, 2: "C"*75, 3: "D"*75}

elif (init_gen_div == 90):
    initial_genomes = {0: "A"*75, 1: "B"*75, 2: "C"*75, 3: "D"*75, 4: "E"*75,
                       5: "F"*75, 6: "G"*75, 7: "H"*75, 8: "I"*75, 9: "J"*75}

dist_humans = {0: 0.0, 1:1.0} 
dist_mosquitoes = {0: 1.0} 
epidemiological_parameters = [sigma_h, gamma, delta, alpha_H, alpha_M, sigma_v, beta_hv, beta_vh]    
population_parameters = {"Mos": num_hum*mos_x_hum , "Hum": num_hum}

iteration_name = f"proof_{iter_number}"
name_fol = f"test/results/Comparative_Simulations/IGD_{init_gen_div}_BR_{num_bites}_NH_{num_hum}_MxH_{mos_x_hum}"

# ------------------------------------------------------------------ #
model = MalariaEGModel(epi_parameters = epidemiological_parameters,
                       pop_parameters = population_parameters,
                       name_folder= name_fol,
                       iteration=iteration_name,
                       distribution=[1/len(initial_genomes[0])] * len(initial_genomes[0]),
                       genomes = initial_genomes,
                       clone_distribution_human = dist_humans,
                       clone_distribution_mosquito = dist_mosquitoes)
# ------------------------------------------------------------------ #
model.run(tmax=365*8)
# ------------------------------------------------------------------ #      