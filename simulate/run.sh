#!/bin/bash

# Nombre del script Python que quieres ejecutar #
PY_SCRIPT="simulate.execute_simulation"
set -e

cd /gsap/garage-protistvector/MalariaKmers/PlasmoEpiGen/

# Activate virtual environment
path_venv="/gsap/garage-protistvector/MalariaKmers/Malaria_Deep-mers/Virtual_Environments/locator_venv"
source "$path_venv/bin/activate"

# Bucle de 0 a 10 #
for i in {0..10}
do
    echo "Ejecución número $i"
    python3 -m $PY_SCRIPT "$i" 10 40
done