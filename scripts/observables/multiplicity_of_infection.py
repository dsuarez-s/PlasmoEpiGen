"""
Observables: Multiplicity of Infection (MOI)
-------------------------------------------
Compute MOI lists for humans and mosquitoes
from the mature_matrix and epidemiological states (X).

Compute MOI values separately for humans and mosquitoes.

Parameters
----------
mature_matrix : sparse or dense matrix (haplotypes x individuals)
    Each row = haplotype, each column = individual.
    Values > 0 mean the haplotype is present in that individual.
X : array-like (n_individuals,)
    Epidemiological state for each individual.
HS, HM, HPC, MS, MC, MPC : int
    State codes for humans (HS, HM, HPC) and mosquitoes (MS, MC, MPC).

Returns
-------
tuple of lists:
    (moi_humans, moi_mosquitoes)
"""

import numpy as np

def measure_moi(mature_matrix, X, HS=0, HM=1, HPC=2, MS=3, MC=4, MPC=5):

    # Boolean presence: haplotypes x individuals
    presence = (mature_matrix > 0).astype(int)

    # Sum across haplotypes -> MOI per individual
    moi_per_ind = np.array(presence.sum(axis=0)).ravel()

    # Partition into humans vs mosquitoes
    humans_mask = np.isin(X, [HM, HPC])
    mosq_mask   = np.isin(X, [MC, MPC])

    if not np.any(humans_mask):
        moi_humans_mean = moi_humans_median = 0.0
    else:
        h = moi_per_ind[humans_mask]
        moi_humans_mean   = float(np.mean(h))
        moi_humans_median = float(np.median(h))

    # Mosquitos
    if not np.any(mosq_mask):
        moi_mosq_mean = moi_mosq_median = 0.0
    else:
        m = moi_per_ind[mosq_mask]
        moi_mosq_mean   = float(np.mean(m))
        moi_mosq_median = float(np.median(m))

    return (round(moi_humans_mean, 2),round(moi_mosq_mean, 2),
            round(moi_humans_median, 2),round(moi_mosq_median, 2))