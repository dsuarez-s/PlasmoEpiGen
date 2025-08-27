import numpy as np
"""
Compute the normalized Shannon diversity index (0-1)
separately for humans and mosquitoes.

Parameters
----------
mature_matrix : np.ndarray or sparse matrix (H x N)
    Rows = haplotypes, Columns = individuals.
    Values >0 indicate presence/abundance.
X : array-like
    Epidemiological states of individuals, used to separate humans vs mosquitoes.

Returns
-------
dict
    {"humans": shannon_value, "mosquitoes": shannon_value}
"""


def measure_shannon_population(mature_matrix, X,
                               HS=0, HM=1, HPC=2,
                               MS=3, MC=4, MPC=5):

    # Identify human and mosquito indices
    human_inds = [i for i, state in enumerate(X) if state in [HS, HM, HPC]]
    mosquito_inds = [i for i, state in enumerate(X) if state in [MS, MC, MPC]]

    results = {}

    for host_type, inds in [("humans", human_inds), ("mosquitoes", mosquito_inds)]:

        # Abundances of haplotypes across this group
        hap_counts = np.array(mature_matrix[:, inds].sum(axis=1)).flatten()
        hap_counts = hap_counts[hap_counts > 0]

        if hap_counts.size == 0:
            raise ValueError(f"No haplotypes present in {host_type} for Shannon calculation")

        # Relative frequencies
        p = hap_counts / hap_counts.sum()

        # Shannon index
        H = -np.sum(p * np.log(p))

        # Normalize (0–1)
        H_max = np.log(len(p))
        if H_max <= 0:
            raise ValueError(f"Invalid haplotype count in {host_type} for Shannon calculation")
        H_norm = float(H / H_max)

        results[host_type] = H_norm

    return results