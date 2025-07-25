import numpy as np

def compute_propensities(X, Hum_Pop, Mos_Pop, 
                         sigma_v, sigma_h, beta_hv, beta_vh, delta, gamma,
                         HS_code, HM_code, HPC_code, MS_code, MC_code, MPC_code):
        
    # División para cada uno de los estados ya sea infección o susceptibilidad a ser infectado #
    inf_mos = (X == MC_code) | (X == MPC_code)
    inf_hum = (X == HM_code) | (X == HPC_code)
    sus_hum = (X == HS_code) | (X == HM_code) | (X == HPC_code)
    sus_mos = (X == MS_code) | (X == MC_code) | (X == MPC_code)
    
    # Calculamos las fuerzas de infección de humanos y vectores #
    prop_bites_h = (sigma_v*Mos_Pop*sigma_h)/(sigma_v*Mos_Pop + sigma_h*Hum_Pop)
    prop_bites_m = (sigma_v*sigma_h*Hum_Pop)/(sigma_v*Mos_Pop + sigma_h*Hum_Pop)

    lambda_h = prop_bites_h*beta_hv*(inf_mos.sum()/Mos_Pop)
    lambda_v = prop_bites_m*beta_vh*(inf_hum.sum()/Hum_Pop)
    
    # Calculamos la propensity para cada uno de los eventos descritos #
    prop_inf_h = lambda_h*sus_hum
    prop_inf_v = lambda_v*sus_mos
    prop_death_mos = (1/delta)*inf_mos
    prop_clearance_hum = (1/gamma)*inf_hum
    
    return np.hstack([prop_inf_h, prop_inf_v, prop_death_mos, prop_clearance_hum])