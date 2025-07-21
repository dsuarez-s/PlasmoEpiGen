def initialize_population_counts(pop_parameters):
    """
    Given pop_parameters, returns a dictionary with population counts:
    - Hum, Mos: total humans and mosquitoes
    - HM_init, HPC_init, HS_init: initial human states
    - MC_init, MPC_init, MS_init: initial mosquito states
    """
    pop = {}
    pop["Hum"] = int(pop_parameters[0])
    pop["Mos"] = int(pop_parameters[1])

    pop["HM_init"]  = int(pop["Hum"] * pop_parameters[2])
    pop["HPC_init"] = int(pop["Hum"] * pop_parameters[3])
    pop["HS_init"]  = pop["Hum"] - pop["HM_init"] - pop["HPC_init"]

    pop["MC_init"]  = int(pop["Mos"] * pop_parameters[4])
    pop["MPC_init"] = int(pop["Mos"] * pop_parameters[5])
    pop["MS_init"]  = pop["Mos"] - pop["MC_init"] - pop["MPC_init"]

    assert pop["HS_init"] >= 0 and pop["MS_init"] >= 0, "Initial population fractions exceed total population."

    return pop