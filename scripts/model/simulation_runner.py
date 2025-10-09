def run(self, tmax):
    # Run the full simulation up to tmax #
    if os.path.isfile(self.path):
        os.remove(self.path)

    ratio = (self.generation_events / self.total_events) if self.total_events > 0 else 0
    self.save_information(0, ratio, len(self.parasitic_populations))
    time_step = 1
    while self.actual_time < tmax:
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
            #print(time_step)
            time_step += 1



    final_results = {"humans_median" : self.observables_humans_median,
                     "humans_mean" : self.observables_humans_mean,
                     "mosquitoes_median" : self.observables_mosquitoes_median,
                     "mosquitoes_mean" : self.observables_mosquitoes_mean}

    # Specify the filename for the pickle file #
    file_dict =  os.path.join(self.config["name_folder"],  "tmp_results_" + str(self.config["iteration"]) + ".pkl")

    # Open the file in binary write mode ('wb')
    with open(file_dict, 'wb') as file:
        # Dump the dictionary into the file
        pickle.dump(final_results, file)