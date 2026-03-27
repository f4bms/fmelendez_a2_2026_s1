import csv
import numpy as np
from scipy import stats
from pathlib import Path
from collections import defaultdict

# =========================
# LECTURA DEL CSV
# =========================

CSV_PATH = Path(__file__).parent / "metrics.csv"

def load_csv(path):
    """
    Devuelve un dict: { (tipo, hilos): [ciclos_simulados, ...] }
    normal siempre tiene hilos=0.
    """
    grupos = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tipo  = row["tipo"]
            hilos = int(row["cant_threads"])
            grupos[(tipo, hilos)].append(int(row["ciclos_simulados"]))
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

OUTPUT_PATH = Path(__file__).parent / "resultados.csv"

FIELDNAMES = [
    "tipo", "hilos", "media_aritmetica", "mediana", "desviacion_estandar",
    "media_geometrica", "IC", "SpeedUp", "Eficiencia"
]

def calcular_fila(tipo, hilos, ciclos, seq_ciclos=None):
    ciclos = np.array(ciclos, dtype=float)
    am  = arithmetic_mean(ciclos)
    med = median(ciclos)
    std = float(np.std(ciclos, ddof=1))
    gm  = geometric_mean(ciclos)
    _, ci = confidence_interval(ciclos)

    if seq_ciclos is not None and hilos > 0:
        n_min = min(len(seq_ciclos), len(ciclos))
        sp  = speedup_am(np.array(seq_ciclos[:n_min]), ciclos[:n_min])
        eff = efficiency(sp, hilos)
    else:
        sp  = 1.0
        eff = ""

    return {
        "tipo":                tipo,
        "hilos":               hilos,
        "media_aritmetica":    round(am,  4),
        "mediana":             round(med, 4),
        "desviacion_estandar": round(std, 4),
        "media_geometrica":    round(gm,  4),
        "IC":                  round(ci,  4),
        "SpeedUp":             round(sp,  4),
        "Eficiencia":          round(eff, 4) if eff != "" else "",
    }

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    grupos = load_csv(CSV_PATH)

    # La baseline secuencial es normal (hilos=0); puede haber varias corridas
    seq_ciclos = grupos.get(("normal", 0), [])

    filas = []

    # Fila para normal
    if seq_ciclos:
        filas.append(calcular_fila("normal", 0, seq_ciclos))

    # Una fila por cada (tipo, hilos) paralelo, ordenado por tipo y luego hilos
    paralelos = sorted(
        ((tipo, hilos) for (tipo, hilos) in grupos if tipo != "normal"),
        key=lambda x: (x[0], x[1])
    )
    for tipo, hilos in paralelos:
        filas.append(calcular_fila(tipo, hilos, grupos[(tipo, hilos)], seq_ciclos))

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(filas)

    print(f"Resultados guardados en: {OUTPUT_PATH}")
    for f in filas:
        print(f"  {f['tipo']:6s} hilos={f['hilos']}  AM={f['media_aritmetica']:,.0f}  SpeedUp={f['SpeedUp']}")
