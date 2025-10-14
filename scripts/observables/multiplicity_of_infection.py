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
    humans_mask = np.isin(X, [HS, HM, HPC])
    mosq_mask   = np.isin(X, [MS, MC, MPC])

    moi_humans_mean = round(np.mean(moi_per_ind[humans_mask].tolist()),2)
    moi_mosq_mean   = round(np.mean(moi_per_ind[mosq_mask].tolist()),2)
    
    moi_humans_median = round(np.median(moi_per_ind[humans_mask].tolist()),2)
    moi_mosq_median   = round(np.median(moi_per_ind[mosq_mask].tolist()),2)

    return moi_humans_mean, moi_mosq_mean, moi_humans_median, moi_mosq_median