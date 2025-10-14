import numpy as np
import itertools
"""
Compute nucleotide diversity (pi) separately for humans and mosquitoes.
Pi is the average pairwise nucleotide difference between haplotypes,
weighted by their abundances (classical definition).

Parameters
----------
mature_matrix : np.ndarray or sparse (H x N)
    Rows = haplotypes, Columns = individuals.
    Values >0 indicate presence/abundance.
X : array-like
    Epidemiological states for individuals (to separate humans vs mosquitoes).
parasitic_populations : array-like
    List of haplotype sequences (strings).

Returns
-------
dict
    {"humans": pi_value, "mosquitoes": pi_value}
"""


# MEASURE_NUCLEOTIDE_DIVERSITY#
def measure_nucleotide_diversity(mature_matrix, X, parasitic_populations,
                                 HS=0, HM=1, HPC=2,
                                 MS=3, MC=4, MPC=5):

    haplo_seqs = {i: np.array(list(seq)) for i, seq in enumerate(parasitic_populations)}
    L = len(parasitic_populations[0])
    results = {}

    for host_type, states in [("humans", [HS, HM, HPC]), ("mosquitoes", [MS, MC, MPC])]:
        inds = [i for i, state in enumerate(X) if state in states]
        if len(inds) == 0:
            raise ValueError(f"No {host_type} found in X")

        # Abundances of haplotypes in this group
        sub_matrix = mature_matrix[:, inds]
        hap_indices = np.flatnonzero(sub_matrix.getnnz(axis=1))
        if hap_indices.size < 2:
            results[host_type] = 0.0
            continue
            
        hap_sum_sparse = sub_matrix.sum(axis=1)
        hap_counts = np.asarray(hap_sum_sparse).ravel()
        
        # Normalize abundances to probabilities
        p = hap_counts[hap_indices] / hap_counts[hap_indices].sum()

        # Pairwise differences weighted by frequencies
        pi = 0.0
        for (i_idx, i), (j_idx, j) in itertools.combinations(enumerate(hap_indices), 2):
            d = np.sum(haplo_seqs[i] != haplo_seqs[j]) / L
            pi += p[i_idx] * p[j_idx] * d

        results[host_type] = round(float(pi * 2), 2) # symmetry correction

    return results["humans"] , results["mosquitoes"]