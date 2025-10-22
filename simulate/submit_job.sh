#!/bin/bash
# Stop on error
set -e

path_script="./run.sh"

# Listas de iteraciones #
lst_A=('10' '2')                                           
lst_B=('0.01' '0.1' '0.5' '1.0' '5.0' '10' '40')           

# Carpetas de logs
log_err="./logs/logs_error"
log_out="./logs/logs_out"
mkdir -p "$log_err" "$log_out"

# Iterar sobre todas las combinaciones posibles
for A in "${lst_A[@]}"; do
    for B in "${lst_B[@]}"; do
        # Etiqueta simple para nombres (reemplaza . por p)
        b_tag="${B//./p}"

        # Limpiar logs anteriores
        err_file="$log_err/${A}_${b_tag}.txt"
        out_file="$log_out/${A}_${b_tag}.txt"
        [ -f "$err_file" ] && rm "$err_file"
        [ -f "$out_file" ] && rm "$out_file"

        # Enviar trabajo al clúster (SGE/UGE)
        qsub -cwd -l h_rt=100:00:00 -l h_vmem=10G -N "Run_${A}_${b_tag}" \
            -e "$err_file" \
            -o "$out_file" \
            "$path_script" "$A" "$B"
    done
done
