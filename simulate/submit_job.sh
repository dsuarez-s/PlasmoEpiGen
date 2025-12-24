#!/bin/bash
# Stop on error
set -e

path_script="./run.sh"

# Simulaciones Diciembre 19 #
lst_init_gen_div=('10' '30' '50' '75' '90')
lst_num_bites=('0.1' '0.2' '0.3' '0.4' '0.6' '0.8' '1.0' '1.5' '2.0' '2.5' '3.0' '3.5' '4.0' '4.5' '5.0')
lst_num_hum=('50')
lst_mos_x_hum=('5')

# Simulaciones Comparacion Humanos #
# lst_init_gen_div=('50' '75' '90')
# lst_num_bites=('2' '5' '10')
# lst_num_hum=('30' '70')
# lst_mos_x_hum=('5')

# Simulaciones Comparacion Mosquitos #
# lst_init_gen_div=('50' '75' '90')
# lst_num_bites=('2' '5' '10')
# lst_num_hum=('50')
# lst_mos_x_hum=('3' '7')

# Carpetas de logs #
log_err="./logs/logs_error"
log_out="./logs/logs_out"
mkdir -p "$log_err" "$log_out"

# Iterar sobre todas las combinaciones posibles #
for init_gen_div in "${lst_init_gen_div[@]}"
do
    for num_bites in "${lst_num_bites[@]}"
    do
        for num_hum in "${lst_num_hum[@]}"
        do 
            for mos_x_hum in "${lst_mos_x_hum[@]}"
            do 
            # Etiqueta simple para nombres (reemplaza . por p) #
            b_num_bites="${num_bites//./p}"

            # Limpiar logs anteriores #
            err_file="$log_err/${init_gen_div}_${b_num_bites}_${num_hum}_${mos_x_hum}.txt"
            out_file="$log_out/${init_gen_div}_${b_num_bites}_${num_hum}_${mos_x_hum}.txt"
            [ -f "$err_file" ] && rm "$err_file"
            [ -f "$out_file" ] && rm "$out_file"

            # Enviar trabajo al clúster # 
            qsub -cwd -l h_rt=100:00:00 -l h_vmem=10G \
            -N "Run_${init_gen_div}_${b_num_bites}_${num_hum}_${mos_x_hum}" \
            -e "$err_file" -o "$out_file" \
            "$path_script" "$init_gen_div" "$num_bites" "$num_hum" "$mos_x_hum"
            done
        done
    done
done
