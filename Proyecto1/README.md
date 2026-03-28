# fmelendez_a2_2026_s1
Proyecto I — Arquitectura II (2026 S1)

## Descripción

Implementación y análisis de rendimiento de una red neuronal feed-forward (3 → 30 → 1) bajo distintos modelos de threading y scheduling.

Se comparan cuatro variantes:

| Variante | Descripción |
|----------|-------------|
| **Normal** | Ejecución secuencial (baseline) |
| **SMT** | Multithreading con scheduling estático por bloques, anclado a cores SMT |
| **SMT-OFF** | Igual que SMT pero con cores físicos distintos (sin compartir recursos) |
| **FGMT** | Fine-Grained Multi-Threading — cambio de contexto tras cada operación |
| **CGMT** | Coarse-Grained Multi-Threading — cambio de contexto solo en stall de caché |

---

## Distribución de archivos

```
Proyecto1/
├── Makefile
├── dataset.txt                        # Dataset de entrenamiento
├── common/
│   ├── nn_utils.h                     # Constantes, activaciones, OpType
│   ├── dataset.h                      # Carga y split train/test
│   └── runner.h                       # Utilidades de impresión
├── NormalExec/
│   ├── main.cpp
│   └── algorythm.cpp                  # Red neuronal secuencial
├── Multithread/
│   ├── mainTh.cpp
│   ├── algorythmTh.cpp                # Red neuronal con block scheduling (SMT/SMT-OFF)
│   ├── fineGrained/
│   │   ├── mainFGMT.cpp
│   │   └── algorythmFGMT.cpp          # Scheduler round-robin + red neuronal FGMT
│   └── coarseGrained/
│       ├── mainCGMT.cpp
│       └── algorythmCGMT.cpp          # Scheduler por stall + red neuronal CGMT
└── results/
    ├── metrics.csv                    # Datos crudos FGMT/CGMT/Normal (simulación)
    ├── smt_th.csv                     # Datos crudos SMT/SMT-OFF (perf stat real)
    ├── resultados.csv                 # Estadísticas FGMT/CGMT/Normal
    ├── resultados_smt.csv             # Estadísticas SMT/SMT-OFF
    ├── metrics.py                     # Calcula estadísticas de metrics.csv
    ├── metrics_smt.py                 # Calcula estadísticas de smt_th.csv
    ├── graficas.py                    # Genera gráficas FGMT/CGMT
    ├── graficas_smt.py                # Genera gráficas SMT/SMT-OFF
    ├── graficas/                      # PNGs de FGMT/CGMT
    └── graficas_smt/                  # PNGs de SMT/SMT-OFF
```

---

## Comandos

### Información del procesador

```bash
# Ver distribución de cores y threads (útil para elegir SMT_CORES)
lscpu -e
```

### Compilar

```bash
cd Proyecto1

make build          # Compila FGMT
make cgmt_build     # Compila CGMT
make normal_build   # Compila Normal (baseline)
make mt_build       # Compila variante SMT/SMT-OFF
```

### Recolectar métricas — simulación (FGMT / CGMT / Normal)

```bash
# Una sola corrida con N hilos
make metrics THREADS=4

# Varias corridas (ej. 30 runs con 4 hilos)
make metrics_runs THREADS=4 RUNS=30

# Generar estadísticas y gráficas
make graficas
```

### Recolectar métricas — perf stat real (SMT / SMT-OFF)

Requiere `perf` y fijar los cores con `taskset`. Ver `lscpu -e` para elegir los cores.

```bash
# SMT: 2 hilos en el mismo core físico (comparten recursos)
make perf_smt SMT_CORES=0,4 THREADS=2 RUNS=30 LABEL=smt

# SMT-OFF: 2 hilos en cores físicos distintos
make perf_smt SMT_CORES=0,1 THREADS=2 RUNS=30 LABEL=smtoff

# Baseline secuencial con perf
make perf_normal NORMAL_CORE=0 RUNS=30 LABEL=normal

# Generar estadísticas y gráficas SMT
make graficas_smt
```

> **Nota:** `SMT_CORES` son los IDs de CPU lógica (columna `CPU` de `lscpu -e`). Para SMT real, elegir dos CPUs con el mismo `CORE` pero distinto `CPU`.

### Limpieza

```bash
make clean          # Elimina binarios y directorio results/
```

---

## Variables del Makefile

| Variable | Default | Descripción |
|----------|---------|-------------|
| `THREADS` | `2` | Número de hilos |
| `RUNS` | `1` | Cantidad de corridas |
| `SEED` | `42` | Semilla del RNG |
| `SMT_CORES` | — | CPUs a usar con `taskset` (ej. `0,4`) |
| `NORMAL_CORE` | `0` | Core para la corrida normal |
| `LABEL` | `sin_label` | Etiqueta `tipo` en el CSV de salida |
| `PYTHON` | `python3` | Intérprete de Python |
