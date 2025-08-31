import numpy as np

"""
Precompute haplotype vs founder IBD values once.

Returns
-------
dict: { founder_name: { hap_index: ibd_value, ... } }
"""

def precompute_ibd_table(mature_matrix, parasitic_populations, genomes):

    founder_lists = {name: np.array(list(name)) for number, name in genomes.items()}

    # Identify active haplotypes
    active_haplos = np.where(mature_matrix.sum(axis=1) > 0)[0]
    haplo_dict = {i: np.array(list(parasitic_populations[i])) for i in active_haplos}
    L = len(parasitic_populations[0])

    ibd_table = {name: {} for name in founder_lists.keys()}

    for h, hap_seq in haplo_dict.items():
        for name, founder_seq in founder_lists.items():
            matches = (hap_seq == founder_seq).sum()
            ibd_table[name][h] = matches / L

    return ibd_table

"""
Compute per-individual IBD relative to each founder using a precomputed table.

Returns
-------
dict: { founder_name: {"humans": [...], "mosquitoes": [...]} }
"""

def measure_ibd_relative_to_founders(mature_matrix, X, ibd_table,
                                     HS=0, HM=1, HPC=2, MS=3, MC=4, MPC=5):

    results = {name: {"humans": [], "mosquitoes": []} for name in ibd_table.keys()}
    
    mature_matrix = mature_matrix.toarray()
    infected_inds = np.where(mature_matrix.sum(axis=0) > 0)[0]

    for ind in infected_inds:
        haplos_present = np.where(mature_matrix[:, ind] > 0)[0]

        for name in ibd_table.keys():
            vals = [ibd_table[name][h] for h in haplos_present]
            avg_ibd = float(np.mean(vals))

            if X[ind] in [HS, HM, HPC]:
                results[name]["humans"].append(round(avg_ibd, 2))
            elif X[ind] in [MS, MC, MPC]:
                results[name]["mosquitoes"].append(round(avg_ibd, 2))

    return results