#!/bin/bash
set -euo pipefail

# Definir las variables a trabajar #
python_script="simulate.execute_simulation"
init_gen_div="$1"   # 10 - 30 - 50 - 75 - 90 #
num_bites="$2"   # 0.6 - 0.7 - 0.8 - 0.9 - 1.0 - 1.2 - 2 - 3 - 10 - 40 #  
num_hum="$3" # 30 - 50 - 70 #
mos_x_hum="$4" # 1 - 3 - 5 #

cd /gsap/garage-protistvector/MalariaKmers/PlasmoEpiGen/

# Activate virtual environment #
path_venv="/gsap/garage-protistvector/MalariaKmers/Malaria_Deep-mers/Virtual_Environments/locator_venv"
source "$path_venv/bin/activate"

# Number of iterations #
total_it=30
for it_num in $(seq 1 "$total_it")
do
    echo "Iteration $it_num of $total_it "
    python3 -m $python_script "$it_num" "$init_gen_div" "$num_bites" "$num_hum" "$mos_x_hum"
done