import numpy as np

def compute_propensities(X, Hum_Pop, Mos_Pop,sigma_v, sigma_h, beta_hv, beta_vh, delta):
    
    # Asignación de los valores para cada categoria #
    HS, HM, HPC = 0, 1, 2
    MS, MC, MPC = 3, 4, 5
    
    # División para cada uno de los estados ya sea infección o susceptibilidad a ser infectado #
    inf_mos = (X == MC) | (X == MPC)
    inf_hum = (X == HM) | (X == HPC)
    sus_hum = (X == HS) | (X == HM) | (X == HPC)
    sus_mos = (X == MS) | (X == MC) | (X == MPC)
    
    # Calculamos las fuerzas de infección de humanos y vectores #
    prop_bites_h = (sigma_v*Mos_Pop*sigma_h)/(sigma_v*Mos_Pop + sigma_h*Hum_Pop)
    prop_bites_m = (sigma_v*sigma_h*Hum_Pop)/(sigma_v*Mos_Pop + sigma_h*Hum_Pop)

    lambda_h = prop_bites_h*beta_hv*(inf_mos.sum()/Mos_Pop)
    lambda_v = prop_bites_m*beta_vh*(inf_hum.sum()/Hum_Pop)
    
    # Calculamos la propensity para cada uno de los eventos descritos #
    prop_inf_h = lambda_h*sus_hum
    prop_inf_v = lambda_v*sus_mos
    prop_death_mos = (1/delta)*inf_mos
    
    return np.hstack([prop_inf_h, prop_inf_v, prop_death_mos])