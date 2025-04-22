# Fast run script for MalariaEGModel #
from scripts.model.malaria_eg_model import MalariaEGModel
import numpy as np

##############################
# Parameters and References ##
##############################

# From gametocytes to sporozoites into mosquitoes #
alpha_M = 14  # days [1]

# From sporozoites to gametocytes into humans #
alpha_H = 11  # days [2]

# From stuffed to hungry / Gonotrophic Cycle #
sigma = 3.1  # days [3]

# Lifespan of sporozoites living inside salivary glands #
xi = 55  # days [4]

# Lifespan of mosquitoes #
mu = 90.4  # days [4]

# Recovery from Infection #
gamma = 289  # days [5]

# Probability of transmission of infection from an infectious mosquito to a susceptible human #
betahv_low = 0.24  # [5]
betahv_high = 0.48  # [5]

# Probability of transmission of infection from an infectious human to a susceptible mosquito #
betavh = 0.022  # [5]

# The maximum number of mosquito bites a human can have per unit time #
val_h_high = 30  # [5]
val_h_low = 4.3  # [5]

# Miscellaneous parameters #
b = 0.36
c = 0.022

# References #

# [1] Dong, Shengzhang, et al. Trends in Parasitology (2021).
# [2] Sauerwein, Meta Roestenberg, Moorthy. Nature Rev Immunol (2011).
# [3] Rúa et al. Memórias Do Instituto Oswaldo Cruz (2005).
# [4] Mayne. Public Health Reports (1922).
# [5] Chitnis, Hyman, Cushing. Bull Math Biol (2008).

#################
# Run the Model #
#################

# Sequences of Genomes #
mos_per_human = 7
Humans = 100
Mosquitoes = Humans * mos_per_human

Percentage_Inf_Hum = 0.5
Percentage_Inf_Mos = 0.5

### Biting Rate ###
a = 40

initial_genomes = {
    0: "AAAAAAAAAAAAAAAAAAAA",
    1: "BBBBBBBBBBBBBBBBBBBB"
}

model = MalariaEGModel(
    epi_parameters=[alpha_H, alpha_M, sigma, a, b, c, xi, mu, gamma],
    pop_parameters=[Humans, Mosquitoes,
                    int(Humans * Percentage_Inf_Hum),
                    int(Mosquitoes * Percentage_Inf_Mos)],
    name_folder="test",
    size_pool=2,
    iteration="proof",
    distribution=[1/20] * 20
)
model.run(tmax=2000, genomes=initial_genomes)
