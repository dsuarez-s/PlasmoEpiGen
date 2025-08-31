# Main Malaria Epidemiological and Genetic Model #

# -------------------------------- #
# Part 1: Imports and Dependencies #
# -------------------------------- #
import numpy as np
import random
import os
from scipy import sparse
import heapq
from collections import defaultdict


# Helpers para populación y genomas #
from scripts.genetics.genome_initialization import initialize_genomes

# Cálculo de tasas #
from scripts.transitions.calculate_forces import compute_propensities

# Funciones de transición de estado #
from scripts.transitions.mosquitoes_events import func_toMS, mosquito_to_human
from scripts.transitions.humans_events import human_to_mosquito, func_toHS
from scripts.transitions.event_queue_schedule import event_queue_execution

# Recombination y limpieza de haplotipos #
from scripts.genetics.recombination import recombination
from scripts.helpers.state_inspectors import update_matrices, classification_S_M_PC

# Metrics measured along the process #
from scripts.observables.identity_by_descent import (precompute_ibd_table, measure_ibd_relative_to_founders as measure_ibd)
from scripts.observables.nucleotide_diversity import (measure_nucleotide_diversity as measure_pi)
from scripts.observables.shannon_index import (measure_shannon_population as measure_shannon)
from scripts.observables.multiplicity_of_infection import measure_moi

"""
sigma_h:   Number of bites per mosquito
gamma:     Recovery time of a human from infection
delta:     Lifespan of mosquitoes
alpha_H:   Maturation time of gametocytes in humans
alpha_M:   Maturation time of sporozoites in mosquitoes
sigma_v:   Rate of gonotrophic cycle (mosquito feeding cycle)
beta_hv:   Probability of transmission from mosquito to human
beta_vh:   Probability of transmission from human to mosquito
xi:        Lifespan of parasites in mosquito salivary glands
"""

# --------------------------------------- #
# Part 2: Model Initialization (__init__) #
# --------------------------------------- #
class MalariaEGModel:
    def __init__(self, epi_parameters, pop_parameters,
                 name_folder, iteration, distribution, genomes,
                 clone_distribution_human, clone_distribution_mosquito):
        
        self.genomes =genomes
        
        self.event_queue = []            
        heapq.heapify(self.event_queue)  
        size_pool = len(genomes)
        # Initialize model parameters #
        self.config = {"name_folder": name_folder,"size_pool": size_pool,
                       "iteration": iteration, "distribution": distribution}
        
        self.observables_humans = {"MOI": [],"IBD": defaultdict(list), "SH": [], "PI": []}
        self.observables_mosquitoes = {"MOI": [],"IBD": defaultdict(list), "SH": [], "PI": []}
        
        # Initialize epidemiological  parameters #
        epi_keys = ["sigma_h", "gamma", "delta", "alpha_H", "alpha_M", "sigma_v", "beta_hv", "beta_vh", "xi"]  
        self.epi = {key: val for key, val in zip(epi_keys, epi_parameters)}   

        # Time and state counters #
        self.actual_time = 0
        self.HS, self.HM, self.HPC = 0, 1, 2
        self.MS, self.MC, self.MPC = 3, 4, 5

        # Initiliaze Populations #  
        
        self.num_mos = pop_parameters["Mos"]
        self.num_hum = pop_parameters["Hum"]        

        
        # 1) Ruta de la carpeta y del archivo
        folder = self.config["name_folder"]
        os.makedirs(folder, exist_ok=True)

        fname = f'BR_{self.epi["beta_hv"]}_IGD_{self.config["size_pool"]}_{self.config["iteration"]}.txt'
        self.path = os.path.join(folder, fname)
        
        # Event list and counters #
        self.events = ["lambda_humans", "lambda_mosquitoes", "toMS", "human_clearance"]
        self.generation_events = 0
        self.total_events = 0
        
        # Genetic initialization #
        init_genomes = initialize_genomes(xi = self.epi["xi"],
                                          genomes_dictionary = genomes,
                                          HM_code = self.HM, HS_code = self.HS, HPC_code = self.HPC,
                                          MC_code = self.MC, MPC_code = self.MPC, MS_code = self.MS,
                                          clone_distribution_human = clone_distribution_human,
                                          num_mos = self.num_mos, num_hum = self.num_hum,
                                          clone_distribution_mosquito = clone_distribution_mosquito,
                                          event_queue = self.event_queue)      
            
            
        self.parasitic_populations = init_genomes[0]
        self.mature_matrix = init_genomes[1]
        self.immature_matrix = init_genomes[2]  
        self.X = init_genomes[3]  
        
        
        
    
    # ------------------------------ #
    # Part 3: Computing Propensities #
    # ------------------------------ #
    
    def calculate_forces(self):
        self.propensities = compute_propensities(X = self.X, Hum_Pop = self.num_hum,Mos_Pop = self.num_mos,
                                                 sigma_v = self.epi["sigma_v"], sigma_h = self.epi["sigma_h"],
                                                 beta_hv = self.epi["beta_hv"], beta_vh = self.epi["beta_vh"],
                                                 delta = self.epi["delta"], gamma = self.epi["gamma"],
                                                 HS_code = self.HS, HM_code = self.HM, HPC_code = self.HPC,
                                                 MS_code = self.MS, MC_code = self.MC, MPC_code = self.MPC)

    # ---------------------------- #
    # Part 4: Selecting Next Event #
    # ---------------------------- #

    def next_time_event(self):
        # Determine next event and time increment #
        total = self.propensities.sum()
        cum = np.cumsum(self.propensities)
        r = np.random.rand()
        self.tau = np.random.exponential(1 / total)
        idx = np.searchsorted(cum, r * total)
        self.transitionPlayer = idx % len(self.X)
        self.transitionType = self.events[idx // len(self.X)]

    # ---------------------------------- #
    # Part 5: Applying State Transitions #
    # ---------------------------------- #
        
    def variate_population(self):
        # Apply state transitions based on event type #
        if self.transitionType == "lambda_humans":
            inoc = mosquito_to_human(X = self.X, 
                                     mature_matrix = self.mature_matrix,
                                     MC_code = self.MC, MPC_code = self.MPC)
            
            for g in inoc:
                if self.immature_matrix[g, self.transitionPlayer] == 0:
                    self.immature_matrix[g, self.transitionPlayer] += 1
                    
                    # Schedule a death event for each assigned haplotype
                    genome_ID = g
                    t_event = self.epi["alpha_H"] + self.actual_time 
                    type_event = "Gametocytes Maturation"
                    agent = self.transitionPlayer
                    heapq.heappush(self.event_queue, (t_event, type_event, genome_ID, agent))

                            
            self.X = classification_S_M_PC(transitionPlayer=self.transitionPlayer,
                                           X_matrix=self.X,
                                           mature_matrix=self.mature_matrix)

        elif self.transitionType == "lambda_mosquitoes":
            inoc = human_to_mosquito(X = self.X,
                                     mature_matrix=self.mature_matrix,
                                     HM_code=self.HM, HPC_code =self.HPC)
                       
            result = recombination(inoculated_genomes=inoc, 
                                   parasitic_populations=self.parasitic_populations,
                                   immature_matrix=self.immature_matrix,
                                   mature_matrix=self.mature_matrix,
                                   total_events=self.total_events,
                                   generation_events=self.generation_events,
                                   dist_loci=self.config["distribution"])
                
            self.parasitic_populations = result[0]
            self.immature_matrix = result[1]
            self.mature_matrix = result[2]
            self.total_events = result[3]
            self.generation_events = result[4]
            selected = result[5]
            
            for g in selected:
                if(self.immature_matrix[g, self.transitionPlayer] == 0):
                    self.immature_matrix[g, self.transitionPlayer] += 1
                    
                    # Schedule a death event for each assigned haplotype #
                    genome_ID = g
                    t_event = self.epi["alpha_M"] + self.actual_time 
                    type_event = "Sporozoites Maturation"
                    agent = self.transitionPlayer
                    heapq.heappush(self.event_queue, (t_event, type_event, genome_ID, agent))
                    
            self.X = classification_S_M_PC(transitionPlayer=self.transitionPlayer,
                                           X_matrix=self.X,
                                           mature_matrix=self.mature_matrix)
            
        elif self.transitionType == "human_clearance":
        
            matrices = func_toHS(transition_Player = self.transitionPlayer,
                                 X_matrix = self.X,
                                 immature_matrix = self.immature_matrix,
                                 mature_matrix = self.mature_matrix,
                                 HS_code = self.HS)
            
            self.X = matrices[0]
            self.mature_matrix = matrices[1]
            self.immature_matrix = matrices[2]
    
            pop_matrix = update_matrices(mature_matrix = self.mature_matrix,
                                         immature_matrix = self.immature_matrix,
                                         parasitic_populations = self.parasitic_populations)

            self.parasitic_populations = pop_matrix[0]
            self.mature_matrix = pop_matrix[1]
            self.immature_matrix = pop_matrix[2]
    
            self.X = classification_S_M_PC(transitionPlayer=self.transitionPlayer,
                                           X_matrix=self.X,
                                           mature_matrix=self.mature_matrix)
        
        else:
            # Mosquito death/reset to susceptible #
            matrices = func_toMS(transition_Player = self.transitionPlayer,
                                 X_matrix = self.X,
                                 immature_matrix = self.immature_matrix,
                                 mature_matrix = self.mature_matrix,
                                 MS_code = self.MS)
            
            self.X = matrices[0]
            self.mature_matrix = matrices[1]
            self.immature_matrix = matrices[2]
    
            pop_matrix = update_matrices(mature_matrix = self.mature_matrix,
                                         immature_matrix = self.immature_matrix,
                                         parasitic_populations = self.parasitic_populations)

            self.parasitic_populations = pop_matrix[0]
            self.mature_matrix = pop_matrix[1]
            self.immature_matrix = pop_matrix[2]
    
            self.X = classification_S_M_PC(transitionPlayer=self.transitionPlayer,
                                           X_matrix=self.X,
                                           mature_matrix=self.mature_matrix)
        
    # -------------------------------- #
    # Part 6: Saving Simulation Output #
    # -------------------------------- #
    def save_information(self, time_step: int, ratio_reco: float, num_haplotypes: int):
        """
        Guarda en un archivo de texto la evolución del sistema.
        - Escribe un encabezado si el archivo no existía.
        - Agrega una línea por cada llamada con:
          time_step;HS;HM;HPC;MS;MC;MPC;ratio_reco;num_haplotypes
        """
                   
        # 2) Encabezado (solo si es la primera vez)
        header = "time;HS;HM;HPC;MS;MC;MPC;ratio_reco;num_haplotypes"
        if not os.path.isfile(self.path):
            with open(self.path, "w") as f:
                f.write(header + "\n")

        # 3) Conteo de cada estado
        nums = [ (self.X == self.HS).sum(), (self.X == self.HM).sum(),
                (self.X == self.HPC).sum(), (self.X == self.MS).sum(),
                (self.X == self.MC).sum(), (self.X == self.MPC).sum()]

        # 4) Línea a registrar
        row = ";".join([str(time_step)] + [str(n) for n in nums] + [f"{ratio_reco}", f"{num_haplotypes}"])

        # 5) Escritura en modo append
        with open(self.path, "a") as f:
            f.write(row + "\n")
            
        self.ibd_table = precompute_ibd_table(self.mature_matrix.toarray(),
                                              self.parasitic_populations,
                                              self.genomes)

        #print("X")
        #print(self.X, self.actual_time)
        moi_h, moi_m = measure_moi(self.mature_matrix, self.X,
                                   HS=self.HS, HM=self.HM, HPC=self.HPC,
                                   MS=self.MS, MC=self.MC, MPC=self.MPC)

        ibd_dict = measure_ibd(self.mature_matrix, self.X, self.ibd_table,
                          HS=self.HS, HM=self.HM, HPC=self.HPC,
                          MS=self.MS, MC=self.MC, MPC=self.MPC)

        sh_dict = measure_shannon(self.mature_matrix, self.X,
                             HS=self.HS, HM=self.HM, HPC=self.HPC,
                             MS=self.MS, MC=self.MC, MPC=self.MPC)

        pi_dict = measure_pi(self.mature_matrix, self.X, self.parasitic_populations,
                        HS=self.HS, HM=self.HM, HPC=self.HPC,
                        MS=self.MS, MC=self.MC, MPC=self.MPC)
        
        self.observables_humans["MOI"].append(list(moi_h))
        self.observables_mosquitoes["MOI"].append(list(moi_m))

        # SH / PI (opcional redondeo)
        sh_h = sh_dict["humans"]
        sh_m = sh_dict["mosquitoes"]
        pi_h = pi_dict["humans"]
        pi_m = pi_dict["mosquitoes"]

        self.observables_humans["SH"].append(sh_h)
        self.observables_mosquitoes["SH"].append(sh_m)
        self.observables_humans["PI"].append(pi_h)
        self.observables_mosquitoes["PI"].append(pi_m)

        # IBD por cepa y host
        for strain, host_vals in ibd_dict.items():
            ih = host_vals["humans"]
            im = host_vals["mosquitoes"]
            self.observables_humans["IBD"][strain].append(ih)        
            self.observables_mosquitoes["IBD"][strain].append(im)    
        #print(self.observables_humans)
    # ---------------------------- #
    # Part 7: Main Simulation Loop #
    # ---------------------------- #
    def run(self, tmax):
        # Run the full simulation up to tmax #
        #self.save_information(0, 0, self.genomes_matrix.shape[0])
        time_step = 1

        if os.path.isfile(self.path):
            os.remove(self.path)
        
        ratio = (self.generation_events / self.total_events) if self.total_events > 0 else 0
        self.save_information(time_step, ratio, len(self.parasitic_populations))
        
        while self.actual_time < tmax:
            print(self.actual_time)
            self.calculate_forces()
            self.next_time_event()
            self.actual_time += self.tau
            
            if(self.event_queue):
                event_queue_executed = event_queue_execution(event_queue = self.event_queue, 
                                                             actual_time = self.actual_time,
                                                             immature_matrix = self.immature_matrix,
                                                             mature_matrix = self.mature_matrix,
                                                             X = self.X, epi_dict = self.epi,
                                                             p_populations =  self.parasitic_populations)

                self.event_queue = event_queue_executed[0]
                self.immature_matrix = event_queue_executed[1]
                self.mature_matrix = event_queue_executed[2]
                self.X = event_queue_executed[3]
                self.parasitic_populations = event_queue_executed[4]
                
            self.variate_population()
            # 2) Guardar stats por cada unidad de tiempo superada

                
            while self.actual_time >= time_step and time_step <= tmax:
                ratio = (self.generation_events / self.total_events) if self.total_events > 0 else 0
                self.save_information(time_step, ratio, len(self.parasitic_populations))

                time_step += 1
                
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        from itertools import chain
        
        def _flatten_ibd_day(ibd_dict, day):
            vals = []
            for _, series in ibd_dict.items():
                if day < len(series):
                    vals.extend(series[day])
            return np.asarray(vals, float)

        def _safe_list_at(L, i):  # para MOI
            return np.asarray(L[i]) if i < len(L) else np.asarray([])

        def _safe_scalar_at(L, i):  # para SH/PI como “1-sample hist”
            return np.asarray([L[i]]) if i < len(L) else np.asarray([])

        def _hist_ylim(samples_iter, bins):
            m = 1
            for arr in samples_iter:
                if arr.size:
                    c, _ = np.histogram(arr, bins=bins)
                    m = max(m, c.max(initial=0))
            return m

        def make_observables_gif(observables_humans, observables_mosquitoes,
                                 out_path="observables.gif", fps=2, dpi=120, alpha=0.6):

            # Número de días
            n_days = max(
                len(observables_humans.get("MOI", [])),
                len(observables_mosquitoes.get("MOI", [])),
                len(observables_humans.get("SH", [])),
                len(observables_mosquitoes.get("SH", [])),
                len(observables_humans.get("PI", [])),
                len(observables_mosquitoes.get("PI", [])),
                max((len(v) for v in observables_humans.get("IBD", {}).values()), default=0),
                max((len(v) for v in observables_mosquitoes.get("IBD", {}).values()), default=0),
            )
            if n_days == 0:
                raise ValueError("No hay datos para animar.")

            # -------- BINS FIJOS --------
            all_moi_h = list(chain.from_iterable(observables_humans.get("MOI", [])))
            all_moi_m = list(chain.from_iterable(observables_mosquitoes.get("MOI", [])))
            moi_max = int(max([0] + all_moi_h + all_moi_m))
            moi_bins = np.arange(0, moi_max + 2)  # bins por entero
            ibd_bins = np.linspace(0, 1, 11)
            sh_bins  = np.linspace(0, 1, 11)
            pi_bins  = np.linspace(0, 1, 11)

            # -------- Y-LIMS FIJOS --------
            moi_samples, ibd_samples, sh_samples, pi_samples = [], [], [], []
            for d in range(n_days):
                moi_samples += [_safe_list_at(observables_humans.get("MOI", []), d),
                                _safe_list_at(observables_mosquitoes.get("MOI", []), d)]
                ibd_samples += [_flatten_ibd_day(observables_humans.get("IBD", {}), d),
                                _flatten_ibd_day(observables_mosquitoes.get("IBD", {}), d)]
                sh_samples  += [_safe_scalar_at(observables_humans.get("SH", []), d),
                                _safe_scalar_at(observables_mosquitoes.get("SH", []), d)]
                pi_samples  += [_safe_scalar_at(observables_humans.get("PI", []), d),
                                _safe_scalar_at(observables_mosquitoes.get("PI", []), d)]

            ylim_moi = _hist_ylim(moi_samples, moi_bins)
            ylim_ibd = _hist_ylim(ibd_samples, ibd_bins)
            ylim_sh  = _hist_ylim(sh_samples,  sh_bins)
            ylim_pi  = _hist_ylim(pi_samples,  pi_bins)

            # -------- FIGURA --------
            fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
            ax_moi, ax_ibd, ax_sh, ax_pi = axes[0,0], axes[0,1], axes[1,0], axes[1,1]

            def draw(ax, data_h, data_m, bins, title, xlabel, ylim):
                ax.cla()
                ax.grid(True, alpha=0.25)
                ax.hist(data_h, bins=bins, alpha=alpha, label="Humans")
                ax.hist(data_m, bins=bins, alpha=alpha, label="Mosquitoes")
                ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Conteo")
                ax.set_xlim(bins[0], bins[-1]); ax.set_ylim(0, ylim)
                ax.legend(loc="upper right")

            def update(day):
                moi_h = _safe_list_at(observables_humans.get("MOI", []), day)
                moi_m = _safe_list_at(observables_mosquitoes.get("MOI", []), day)
                ibd_h = _flatten_ibd_day(observables_humans.get("IBD", {}), day)
                ibd_m = _flatten_ibd_day(observables_mosquitoes.get("IBD", {}), day)
                sh_h  = _safe_scalar_at(observables_humans.get("SH", []), day)
                sh_m  = _safe_scalar_at(observables_mosquitoes.get("SH", []), day)
                pi_h  = _safe_scalar_at(observables_humans.get("PI", []), day)
                pi_m  = _safe_scalar_at(observables_mosquitoes.get("PI", []), day)

                draw(ax_moi, moi_h, moi_m, moi_bins, f"MOI — Día {day}", "MOI", ylim_moi)
                draw(ax_ibd, ibd_h, ibd_m, ibd_bins, f"IBD — Día {day}", "IBD", ylim_ibd)
                draw(ax_sh,  sh_h,  sh_m,  sh_bins,  f"SH — Día {day}",  "SH",  ylim_sh)
                draw(ax_pi,  pi_h,  pi_m,  pi_bins,  f"PI — Día {day}",  "PI",  ylim_pi)
                fig.suptitle(f"Observables — Día {day}", fontsize=14)
                return axes.ravel()

            anim = FuncAnimation(fig, update, frames=n_days, interval=1000//fps, blit=False)

            # Guardar como GIF (no requiere ffmpeg)
            writer = PillowWriter(fps=fps)
            anim.save(out_path, writer=writer, dpi=dpi)
            plt.close(fig)
            print(f"[OK] GIF guardado en: {out_path}")
            return out_path


        make_observables_gif(self.observables_humans, self.observables_mosquitoes, out_path="observables.gif", fps=2, dpi=120)