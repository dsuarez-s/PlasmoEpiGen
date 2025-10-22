#!/bin/bash

# Nombre del script Python que quieres ejecutar #
PY_SCRIPT="simulate.execute_simulation"
set -e
A="$1"   # 10 o 2
B="$2"   # 0.01 ... 40

cd /gsap/garage-protistvector/MalariaKmers/PlasmoEpiGen/

# Activate virtual environment
path_venv="/gsap/garage-protistvector/MalariaKmers/Malaria_Deep-mers/Virtual_Environments/locator_venv"
source "$path_venv/bin/activate"

# Bucle de 0 a 10 #
for i in {0..10}
do
    echo "Ejecución número $i"
    python3 -m $PY_SCRIPT "$i" "$A" "$B"
done