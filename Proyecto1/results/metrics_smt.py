import csv
import numpy as np
from scipy import stats
from pathlib import Path
from collections import defaultdict

# =========================
# LECTURA DEL CSV
# =========================

CSV_PATH    = Path(__file__).parent / "smt_th.csv"
OUTPUT_PATH = Path(__file__).parent / "resultados_smt.csv"

def load_csv(path):
    """
    Devuelve un dict: { (tipo, threads): [task_clock_s, ...] }
    tipo=normal actúa como baseline secuencial, igual que en metrics.py.
    """
    grupos = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tipo    = row["tipo"]
            threads = int(row["threads"])
            grupos[(tipo, threads)].append(float(row["task-clock (s)"]))
    return grupos

# =========================
# FUNCIONES ESTADÍSTICAS
# =========================

def arithmetic_mean(data):
    return np.mean(data)

def median(data):
    return np.median(data)

def geometric_mean(data):
    return np.exp(np.mean(np.log(np.array(data, dtype=float))))

def confidence_interval(data, confidence=0.95):
    data = np.array(data, dtype=float)
    n    = len(data)
    mean = np.mean(data)
    std  = np.std(data, ddof=1)
    h    = stats.t.ppf((1 + confidence) / 2, n - 1) * (std / np.sqrt(n))
    return mean, h

def speedup_am(seq, par):
    return np.mean(seq) / np.mean(par)

def efficiency(speedup, n_threads):
    return speedup / n_threads

# =========================
# GUARDAR RESULTADOS
# =========================

FIELDNAMES = [
    "tipo", "hilos", "media_aritmetica", "mediana", "desviacion_estandar",
    "media_geometrica", "IC", "SpeedUp", "Eficiencia"
]

def calcular_fila(tipo, hilos, tiempos, seq_tiempos=None):
    tiempos = np.array(tiempos, dtype=float)
    am  = arithmetic_mean(tiempos)
    med = median(tiempos)
    std = float(np.std(tiempos, ddof=1))
    gm  = geometric_mean(tiempos)
    _, ci = confidence_interval(tiempos)

    # SpeedUp = t_normal / t_paralelo (misma lógica que metrics.py con ciclos)
    if seq_tiempos is not None and hilos > 0:
        n_min = min(len(seq_tiempos), len(tiempos))
        sp  = speedup_am(np.array(seq_tiempos[:n_min]), tiempos[:n_min])
        eff = efficiency(sp, hilos)
    else:
        sp  = 1.0
        eff = ""

    return {
        "tipo":                tipo,
        "hilos":               hilos,
        "media_aritmetica":    round(am,  6),
        "mediana":             round(med, 6),
        "desviacion_estandar": round(std, 6),
        "media_geometrica":    round(gm,  6),
        "IC":                  round(ci,  6),
        "SpeedUp":             round(sp,  4),
        "Eficiencia":          round(eff, 4) if eff != "" else "",
    }

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    grupos = load_csv(CSV_PATH)

    # Baseline: tipo=normal, igual que metrics.py usa ("normal", 0)
    seq_tiempos = grupos.get(("normal", 1), []) or grupos.get(("normal", 0), [])

    filas = []

    # Fila para normal
    if seq_tiempos:
        filas.append(calcular_fila("normal", list({t for (tp, t) in grupos if tp == "normal"})[0], seq_tiempos))

    # Resto de tipos ordenados
    paralelos = sorted(
        ((tipo, hilos) for (tipo, hilos) in grupos if tipo != "normal"),
        key=lambda x: (x[0], x[1])
    )
    for tipo, hilos in paralelos:
        filas.append(calcular_fila(tipo, hilos, grupos[(tipo, hilos)], seq_tiempos))

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(filas)

    print(f"Resultados guardados en: {OUTPUT_PATH}")
    for f in filas:
        print(f"  {f['tipo']:12s} hilos={f['hilos']}  AM={f['media_aritmetica']:.6f}s  SpeedUp={f['SpeedUp']}")
