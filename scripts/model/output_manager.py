import os
import numpy as np
from scripts.observables.identity_by_descent import (precompute_ibd_table, measure_ibd_relative_to_founders as measure_ibd)
from scripts.observables.nucleotide_diversity import (measure_nucleotide_diversity as measure_pi)
from scripts.observables.shannon_index import ( measure_shannon_population as measure_shannon)
from scripts.observables.multiplicity_of_infection import measure_moi

# Función 1: Convierte a string redondeado a 2 decimales o 'nan' si el valor no es computable #
def to_str_or_nan(value):
    if value is None:
        return "nan"
    try:
        return str(np.round(value, 2))
    except Exception:
        return "nan"

# Función 2: Guarda la información de la simulación en el tiempo requerido #
def save_information(self, time_step):
    
    # Métrica 1: Calcula la proporción de eventos recombinantes respecto a todos los eventos de infección #
    ratio_reco = (self.infect_with_reco / self.total_events_infect) if self.total_events_infect > 0 else 0
    
    # Métrica 2: Número de haplotipos circulantes en la población #
    alive_m = np.asarray(self.mature_matrix.getnnz(axis=1)).ravel()
    alive_i = np.asarray(self.immature_matrix.getnnz(axis=1)).ravel()
    alive = (alive_m + alive_i) > 0 
    num_haplotypes = alive.sum()
    
    # Métrica 3: Fuerza de infección anual para mosquitos y humanos #
    # Humanos #
    inf_mos = (self.X == self.MC) | (self.X == self.MPC)
    prop_bites_h_num = (self.epi["sigma_v"] * self.num_mos * self.epi["sigma_h"])
    prop_bites_h_den =(self.epi["sigma_v"] * self.num_mos + self.epi["sigma_h"] * self.num_hum)
    prop_bites_h = prop_bites_h_num/prop_bites_h_den
    lambda_h = np.round(365*prop_bites_h * self.epi["beta_hv"] * (inf_mos.sum() / self.num_mos),2)
    # Mosquitos #
    inf_hum = (self.X == self.HM) | (self.X == self.HPC)
    prop_bites_m_num = (self.epi["sigma_v"] * self.epi["sigma_h"] * self.num_hum)
    prop_bites_m_den = (self.epi["sigma_v"] * self.num_mos + self.epi["sigma_h"] * self.num_hum)
    prop_bites_m = prop_bites_m_num/prop_bites_m_den
    lambda_v = np.round(365*prop_bites_m * self.epi["beta_vh"] * (inf_hum.sum() / self.num_hum),2)

    # Métrica 4: Conteo del estado epidemiológico de cada agente #
    nums = [(self.X == self.HS).sum(), (self.X == self.HM).sum(),
            (self.X == self.HPC).sum(), (self.X == self.MS).sum(),
            (self.X == self.MC).sum(), (self.X == self.MPC).sum()]

    # Métrica 5: MOI para humanos y mosquitos #
    moi_h_mean, moi_m_mean, moi_h_median, moi_m_median = measure_moi(self.mature_matrix, self.X,
                                                                     HS=self.HS, HM=self.HM, HPC=self.HPC,
                                                                     MS=self.MS, MC=self.MC, MPC=self.MPC)

    # Métrica 6: Diversidad nucleotídica poblacional (π) en humanos y mosquitos #
    pi_humans, pi_mosquitoes = measure_pi(self.mature_matrix, self.X, self.parasitic_populations,
                                          HS=self.HS, HM=self.HM, HPC=self.HPC,
                                          MS=self.MS, MC=self.MC, MPC=self.MPC)

    # Métrica 7: Índice de Shannon normalizado en humanos y mosquitos #
    sh_humans, sh_mosquitoes = measure_shannon(self.mature_matrix, self.X,
                                               HS=self.HS, HM=self.HM, HPC=self.HPC,
                                               MS=self.MS, MC=self.MC, MPC=self.MPC)

   # Check 1: Si el archivo no existe se crea uno vacios con los encabezados de las métricas separados por ; #
    if not os.path.isfile(self.path):
        header_parts = ["time", "HS", "HM", "HPC", "MS", "MC", "MPC",
                        "ratio_reco", "num_haplotypes",
                        "MOI_Humans_mean", "MOI_Humans_median",
                        "MOI_Mosquitoes_mean", "MOI_Mosquitoes_median",
                        "SH_Humans", "SH_Mosquitoes", "PI_Humans", "PI_Mosquitoes","lambda_h", "lambda_v"]
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            f.write(";".join(header_parts) + "\n")
            
            
    # Paso 1: Se agrega una fila con los datos de cada métrica #
    row_parts = ([to_str_or_nan(time_step)] + [str(n) for n in nums] 
                 + [to_str_or_nan(ratio_reco), str(num_haplotypes)]
                 + [str(moi_h_mean), str(moi_h_median), str(moi_m_mean), str(moi_m_median)]
                 + [str(sh_humans), str(sh_mosquitoes), str(pi_humans), str(pi_mosquitoes)]
                 + [str(lambda_h),str(lambda_v)])
    with open(self.path, "a", encoding="utf-8", newline="") as f:
        f.write(";".join(row_parts) + "\n")
        
        
# CALCULO INCLUYENDO IBD #
#     # Precómputo de tabla IBD para acelerar cálculos posteriores #
#     self.ibd_table = precompute_ibd_table(self.mature_matrix,
#                                           self.parasitic_populations,
#                                           self.genomes)

#     # Medición de IBD relativo a fundadores por hospedador y por cepa #
#     ibd_dict = measure_ibd(self.mature_matrix, self.X, self.ibd_table,
#                            HS=self.HS, HM=self.HM, HPC=self.HPC,
#                            MS=self.MS, MC=self.MC, MPC=self.MPC)

#     # Acumuladores de métricas IBD por cepa (listas fuera del bucle) #
#     ls_h_mean, ls_h_median, ls_m_mean, ls_m_median = [], [], [], []
#     strains = sorted(ibd_dict.keys())  # Orden de columnas IBD consistente con los valores #
    
#     for strain, host_vals in ibd_dict.items():
#         # Extrae arrays por hospedador, usa lista vacía si falta la clave #
#         ih = host_vals.get("humans", [])
#         im = host_vals.get("mosquitoes", [])

#         # Cálculos robustos a listas vacías (devuelven 'nan' si no hay datos) #
#         h_mean = to_str_or_nan(np.mean(ih) if len(ih) else None)
#         h_median = to_str_or_nan(np.median(ih) if len(ih) else None)
#         m_mean = to_str_or_nan(np.mean(im) if len(im) else None)
#         m_median = to_str_or_nan(np.median(im) if len(im) else None)

#         # Agregado por cepa en el mismo orden en que generaremos el encabezado #
#         ls_h_mean.append(h_mean)
#         ls_h_median.append(h_median)
#         ls_m_mean.append(m_mean)
#         ls_m_median.append(m_median)

#     # Check 1: Si el archivo no existe se crea uno nuevo con #
#     if not os.path.isfile(self.path):
#         header_parts = ["time", "HS", "HM", "HPC", "MS", "MC", "MPC",
#                         "ratio_reco", "num_haplotypes",
#                         "MOI_Humans_mean", "MOI_Humans_median",
#                         "MOI_Mosquitoes_mean", "MOI_Mosquitoes_median",
#                         "SH_Humans", "SH_Mosquitoes", "PI_Humans", "PI_Mosquitoes","lambda_h", "lambda_v"]
        
#         # Campos IBD por cepa, agrupados por métrica para facilitar lectura #
#         header_parts += [f"h_mean_{s}"   for s in self.initial_genomes]
#         header_parts += [f"h_median_{s}" for s in self.initial_genomes]
#         header_parts += [f"m_mean_{s}"   for s in self.initial_genomes]
#         header_parts += [f"m_median_{s}" for s in self.initial_genomes]

#         # Escritura del encabezado con separador ';' una sola vez (evita separadores faltantes) #
#         with open(self.path, "w", encoding="utf-8", newline="") as f:
#             f.write(";".join(header_parts) + "\n")

#     # Fila de datos (mismo orden que el encabezado) #
#     row_parts = ([to_str_or_nan(time_step)] + [str(n) for n in nums] 
#                  + [to_str_or_nan(ratio_reco), str(num_haplotypes)]
#                  + [str(moi_h_mean), str(moi_h_median), str(moi_m_mean), str(moi_m_median)]
#                  + [str(sh_humans), str(sh_mosquitoes), str(pi_humans), str(pi_mosquitoes)]
#                  + [str(lambda_h),str(lambda_v)] 
#                  + ls_h_mean + ls_h_median + ls_m_mean + ls_m_median)   
#     Escritura en modo append, garantizando separadores correctos #
#     with open(self.path, "a", encoding="utf-8", newline="") as f:
#         f.write(";".join(row_parts) + "\n")