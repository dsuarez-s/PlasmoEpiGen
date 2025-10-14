import numpy as np
from typing import Dict, Any
from scipy.sparse import csr_matrix

def validate_inputs_ibd(mature_matrix, parasitic_populations, genomes):
    # Tipo CSR
    if not isinstance(mature_matrix, csr_matrix):
        raise TypeError("mature_matrix debe ser scipy.sparse.csr_matrix")

    # parasitic_populations: ndarray 1D de strings no vacío
    if not isinstance(parasitic_populations, np.ndarray):
        raise TypeError("parasitic_populations debe ser un np.ndarray")
    if parasitic_populations.ndim != 1:
        raise ValueError("parasitic_populations debe ser 1D con shape (n_haplos,)")
    if parasitic_populations.size == 0:
        raise ValueError("parasitic_populations no puede estar vacío")

    n_haplos = parasitic_populations.shape[0]
    try:
        L = len(parasitic_populations[0])
    except Exception:
        raise TypeError("Cada elemento de parasitic_populations (1D) debe ser una secuencia iterable")
    if L == 0:
        raise ValueError("Las secuencias de parasitic_populations no pueden tener longitud 0")

    # Todas las secuencias deben tener la misma longitud
    for i, s in enumerate(parasitic_populations):
        if len(s) != L:
            raise ValueError(f"Las secuencias deben tener longitud uniforme; índice {i} tiene {len(s)} != {L}")

    # Coincidencia de filas con la CSR
    if mature_matrix.shape[0] != n_haplos:
        raise ValueError(
            f"Inconsistencia: mature_matrix tiene {mature_matrix.shape[0]} filas "
            f"pero parasitic_populations tiene {n_haplos}")

    # genomes: dict no vacío con secuencias iterables y de longitud L
    if not isinstance(genomes, dict):
        raise TypeError("genomes debe ser un dict {nombre: secuencia}")
    if len(genomes) == 0:
        raise ValueError("genomes no puede estar vacío")

    for name, seq in genomes.items():
        if seq is None:
            raise ValueError(f"La secuencia del fundador '{name}' es None")
        try:
            seq_list = list(seq)  # iterable (str/list/np.array)
        except TypeError:
            raise TypeError(f"La secuencia del fundador '{name}' debe ser iterable (str, lista, np.array)")
        if len(seq_list) != L:
            raise ValueError(f"La secuencia del fundador '{name}' tiene longitud {len(seq_list)} != {L}")

    return L

def precompute_ibd_table( mature_matrix: csr_matrix, parasitic_populations: np.ndarray,
                         genomes: Dict[Any, str]) -> Dict[str, Dict[int, float]]:
    
    print(parasitic_populations)
    # --- Validaciones clave #
    L = validate_inputs_ibd(mature_matrix, parasitic_populations, genomes)
    
    # Fundadores -> arrays #
    founder_lists = {name: np.array(list(seq)) for name, seq in genomes.items()}
    # Haplótipos activos (CSR) #
    active_haplos = np.flatnonzero(mature_matrix.getnnz(axis=1))
    print("active_haplos", active_haplos)
    # Si no hay activos, devolver diccionario vacío por fundador
    if active_haplos.size == 0:
        return {name: {} for name in founder_lists.keys()}

    # Construcción de la tabla
    ibd_table = {name: {} for name in founder_lists.keys()}
    for h in active_haplos:
        hap_seq = np.array(list(parasitic_populations[h]))
        for name, founder_seq in founder_lists.items():
            matches = (hap_seq == founder_seq).sum()
            ibd_table[name][int(h)] = round(matches / L,2)
    print(ibd_table)
    return ibd_table


def measure_ibd_relative_to_founders(mature_matrix: csr_matrix,X: np.ndarray,
                                     ibd_table: Dict[str, Dict[int, float]],
                                     HS=0, HM=1, HPC=2, MS=3, MC=4, MPC=5) -> Dict[str, Dict[str, list]]:
    # Resultados por fundador
    results = {name: {"humans": [], "mosquitoes": []} for name in ibd_table.keys()}

    # 1) Columnas infectadas (individuos con ≥1 haplotipo) sin densificar #
    infected_inds = np.flatnonzero(mature_matrix.getnnz(axis=0))

    # 2) Para leer eficientemente índices de filas por columna, usa CSC #
    M_csc = mature_matrix.tocsc()

    # 3) Recorremos solo individuos infectados #
    indptr = M_csc.indptr   # punteros por columna
    indices = M_csc.indices # filas no-cero (haplotipos) alineados a indptr #

    for ind in infected_inds:
        # Haplotipos presentes en la columna 'ind' (sin slicing costoso) #
        start, end = indptr[ind], indptr[ind + 1]
        haplos_present = indices[start:end]  # np.ndarray de ints (>=1 elemento)

        for name in ibd_table.keys():
            # Tomamos valores existentes; si algún h no está, lo ignoramos #
            vals = [ibd_table[name][h] for h in haplos_present if h in ibd_table[name]]
            if not vals:
                # Si por alguna razón no hay valores (p.ej., filtro previo), saltamos
                continue

            avg_ibd = float(np.mean(vals))
            if X[ind] in (HS, HM, HPC):
                results[name]["humans"].append(round(avg_ibd, 2))
            elif X[ind] in (MS, MC, MPC):
                results[name]["mosquitoes"].append(round(avg_ibd, 2))

    return results