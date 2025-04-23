import numpy as np

def compute_propensities(X, Hum_Pop, Mos_Pop, sigma, a, b, c, mu, HM, HPC, HS, MC, MPC, MS):
    sigma_v = 1 / sigma

    active_mos = (X == MC) | (X == MPC)
    inf_hum = (X == HM) | (X == HPC)
    sus_hum = (X == HS) | inf_hum

    prop_bites_h = (sigma_v * Mos_Pop * a) / (
        sigma_v * Mos_Pop + a * Hum_Pop
    )
    lambda_humans = prop_bites_h * b * (active_mos.sum() / Mos_Pop) * sus_hum

    sus_mos = (X == MS) | (X == MC) | (X == MPC)
    prop_bites_m = (sigma_v * a * Hum_Pop) / (
        sigma_v * Mos_Pop + a * Hum_Pop
    )
    lambda_mosquitoes = prop_bites_m * c * (inf_hum.sum() / Hum_Pop) * sus_mos

    toMS = (1 / mu) * active_mos

    return np.hstack([lambda_humans, lambda_mosquitoes, toMS])