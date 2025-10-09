def save_information(self, time_step: int, ratio_reco: float, num_haplotypes: int):

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

    # Análisis de Observables #    
    self.ibd_table = precompute_ibd_table(self.mature_matrix.toarray(),
                                          self.parasitic_populations,
                                          self.genomes)

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

    self.observables_humans_median["MOI"].append(np.round(np.median(moi_h),2))
    self.observables_humans_mean["MOI"].append(np.round(np.mean(moi_h),2))

    self.observables_mosquitoes_median["MOI"].append(np.round(np.median(moi_m),2))
    self.observables_mosquitoes_mean["MOI"].append(np.round(np.mean(moi_m),2))

    # SH / PI (opcional redondeo)
    sh_h = sh_dict["humans"]
    sh_m = sh_dict["mosquitoes"]
    pi_h = pi_dict["humans"]
    pi_m = pi_dict["mosquitoes"]

    self.observables_humans_median["SH"].append(np.round(np.median(sh_h),2))
    self.observables_humans_mean["SH"].append(np.round(np.mean(sh_h),2))

    self.observables_mosquitoes_median["SH"].append(np.round(np.median(sh_m),2))
    self.observables_mosquitoes_mean["SH"].append(np.round(np.mean(sh_m),2))

    self.observables_humans_median["PI"].append(np.round(np.median(pi_h),2))
    self.observables_humans_mean["PI"].append(np.round(np.mean(pi_h),2))

    self.observables_mosquitoes_median["PI"].append(np.round(np.median(pi_m),2))
    self.observables_mosquitoes_mean["PI"].append(np.round(np.mean(pi_m),2))

    # IBD por cepa y host #
    for strain, host_vals in ibd_dict.items():
        ih = host_vals["humans"]
        im = host_vals["mosquitoes"]
        self.observables_humans_median["IBD"][strain].append(np.round(np.median(ih),2))
        self.observables_humans_mean["IBD"][strain].append(np.round(np.mean(ih),2))

        self.observables_mosquitoes_median["IBD"][strain].append(np.round(np.median(im),2))
        self.observables_mosquitoes_mean["IBD"][strain].append(np.round(np.mean(im),2))