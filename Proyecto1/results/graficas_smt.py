import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH     = Path(__file__).parent / "resultados_smt.csv"
METRICS_PATH = Path(__file__).parent / "smt_th.csv"
OUT_DIR      = Path(__file__).parent / "graficas_smt"
OUT_DIR.mkdir(exist_ok=True)

import csv as _csv

df  = pd.read_csv(CSV_PATH)

# Leer smt_th.csv con soporte para filas de 9 y 12 columnas.
# smt/smtoff: 9 cols (sin branch-misses/ctx-switches/cpu-migrations)
# normal/cgmt: 12 cols (con esos campos extra)
# csv.reader maneja los campos entre comillas con comas internas (ej. "(0,1)").
_rows = []
with open(METRICS_PATH, newline="") as _f:
    _reader = _csv.reader(_f)
    next(_reader)  # saltar header
    for _p in _reader:
        if len(_p) < 5 or not _p[4].strip():
            continue
        try:
            _rows.append({
                "tipo":           _p[0],
                "threads":        int(_p[1]),
                "run":            int(_p[3]),
                "task-clock (s)": float(_p[4]),
                "cycles":         int(_p[5]) if len(_p) > 5 and _p[5].strip() else None,
                "instructions":   int(_p[6]) if len(_p) > 6 and _p[6].strip() else None,
                "cache-misses":   int(_p[7]) if len(_p) > 7 and _p[7].strip() else None,
            })
        except (ValueError, IndexError):
            pass
raw = pd.DataFrame(_rows)

# N por grupo
n_counts = raw.groupby(["tipo", "threads"]).size()

def get_n(tipo, threads):
    return int(n_counts.get((tipo, threads), 0))

n_smtoff = get_n("smtoff", 2)

smt    = df[df["tipo"] == "smt"].sort_values("hilos")
smtoff = df[df["tipo"] == "smtoff"].sort_values("hilos")
cgmt   = df[df["tipo"] == "cgmt"].sort_values("hilos")
normal = df[df["tipo"] == "normal"].iloc[0]

# -- Tiempo de ejecución vs Hilos – SMT vs SMT-OFF vs CGMT (comparado) --------
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(smt["hilos"], smt["media_aritmetica"],
        marker="o", color="#2196F3", linewidth=2, markersize=8, label="SMT")
ax.fill_between(
    smt["hilos"],
    smt["media_aritmetica"] - smt["IC"],
    smt["media_aritmetica"] + smt["IC"],
    alpha=0.15, color="#2196F3", label="IC 95% SMT"
)
ax.plot(smtoff["hilos"], smtoff["media_aritmetica"],
        marker="s", color="#FF9800", linewidth=0, markersize=10, label="SMT-OFF")
ax.errorbar(smtoff["hilos"], smtoff["media_aritmetica"],
            yerr=smtoff["IC"],
            fmt="none", ecolor="#FF9800", capsize=6, capthick=1.5)
if not cgmt.empty:
    ax.plot(cgmt["hilos"], cgmt["media_aritmetica"],
            marker="^", color="#4CAF50", linewidth=2, markersize=8, label="CGMT")
    ax.fill_between(
        cgmt["hilos"],
        cgmt["media_aritmetica"] - cgmt["IC"],
        cgmt["media_aritmetica"] + cgmt["IC"],
        alpha=0.15, color="#4CAF50", label="IC 95% CGMT"
    )
ax.set_xlabel("Hilos")
ax.set_ylabel("Tiempo de ejecución (s)")
ax.set_title("Tiempo de Ejecución vs Hilos – SMT vs SMT-OFF vs CGMT")
all_hilos = smt["hilos"].tolist() + smtoff["hilos"].tolist() + (cgmt["hilos"].tolist() if not cgmt.empty else [])
ax.set_xticks(sorted(set(all_hilos)))
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.savefig(OUT_DIR / "tiempo_comparado.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  guardada: tiempo_comparado.png")

# -- Eficiencia vs Hilos – SMT vs SMT-OFF vs CGMT (comparado) -----------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(smt["hilos"], smt["Eficiencia"],
        marker="o", color="#2196F3", linewidth=2, markersize=8, label="SMT")
ax.plot(smtoff["hilos"], smtoff["Eficiencia"],
        marker="s", color="#FF9800", linewidth=0, markersize=10, label="SMT-OFF")
if not cgmt.empty:
    ax.plot(cgmt["hilos"], cgmt["Eficiencia"],
            marker="^", color="#4CAF50", linewidth=2, markersize=8, label="CGMT")
ax.set_xlabel("Hilos")
ax.set_ylabel("Eficiencia")
ax.set_title("Eficiencia vs Hilos – SMT vs SMT-OFF vs CGMT")
all_hilos = smt["hilos"].tolist() + smtoff["hilos"].tolist() + (cgmt["hilos"].tolist() if not cgmt.empty else [])
ax.set_xticks(sorted(set(all_hilos)))
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.savefig(OUT_DIR / "eficiencia_comparado.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  guardada: eficiencia_comparado.png")

# -- SpeedUp vs Hilos – SMTOFF (barra) ----------------------------------------
fig, ax = plt.subplots(figsize=(4, 4))
ax.bar("SMT-OFF\n2 hilos", smtoff["SpeedUp"].iloc[0],
       color="#FF9800", width=0.4, label="SMT-OFF")
ax.set_ylabel("SpeedUp")
ax.set_title(f"SpeedUp – SMT-OFF (N={n_smtoff})")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.savefig(OUT_DIR / "speedup_smtoff.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  guardada: speedup_smtoff.png")

# -- SpeedUp comparado: SMT vs SMT-OFF vs CGMT --------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(smt["hilos"], smt["SpeedUp"],
        marker="o", color="#2196F3", linewidth=2, markersize=8, label="SMT")
ax.plot(smtoff["hilos"], smtoff["SpeedUp"],
        marker="s", color="#FF9800", linewidth=0, markersize=10, label="SMT-OFF")
if not cgmt.empty:
    ax.plot(cgmt["hilos"], cgmt["SpeedUp"],
            marker="^", color="#4CAF50", linewidth=2, markersize=8, label="CGMT")
ax.set_xlabel("Hilos")
ax.set_ylabel("SpeedUp")
ax.set_title("SpeedUp vs Hilos – SMT vs SMT-OFF vs CGMT")
all_hilos = smt["hilos"].tolist() + smtoff["hilos"].tolist() + (cgmt["hilos"].tolist() if not cgmt.empty else [])
ax.set_xticks(sorted(set(all_hilos)))
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.savefig(OUT_DIR / "speedup_comparado.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  guardada: speedup_comparado.png")

# -- Boxplot distribución de tiempos – SMT, SMT-OFF y CGMT -------------------
COLORES = {"smt": "#90CAF9", "smtoff": "#FFCC80", "cgmt": "#A5D6A7"}

for tipo in ("smt", "smtoff", "cgmt"):
    grupo_tipo = raw[raw["tipo"] == tipo]
    grupos_box = grupo_tipo.groupby("threads")["task-clock (s)"]
    data_box   = [g.values for _, g in grupos_box]
    labels_box = [f"{h} hilo{'s' if h != 1 else ''}" for h, _ in grupos_box]

    if not data_box:
        continue
    fig, ax = plt.subplots(figsize=(max(5, len(labels_box) * 1.6), 5))
    ax.boxplot(data_box, tick_labels=labels_box, patch_artist=True,
               boxprops=dict(facecolor=COLORES[tipo], color="#333333"),
               medianprops=dict(color="#E53935", linewidth=2))
    ax.set_ylabel("Tiempo de ejecución (s)")
    ax.set_title(f"Distribución de tiempos – {tipo.upper()}")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fname = f"boxplot_{tipo}.png"
    fig.savefig(OUT_DIR / fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  guardada: {fname}")

print(f"\nGráficas guardadas en: {OUT_DIR}/")

# -- Ciclos por iteración (una gráfica por tipo) -------------------------------
for tipo, grupo_tipo in raw.groupby("tipo"):
    fig, ax = plt.subplots(figsize=(8, 4))
    for threads, grupo_th in grupo_tipo.groupby("threads"):
        grupo_th = grupo_th.sort_values("run")
        ax.plot(grupo_th["run"], grupo_th["cycles"],
                marker="o", markersize=4, linewidth=1.5,
                label=f"{threads} hilo{'s' if threads != 1 else ''}")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Ciclos")
    ax.set_title(f"Variación de ciclos por iteración – {tipo.upper()}")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.5)
    fig.tight_layout()
    fname = f"ciclos_{tipo}.png"
    fig.savefig(OUT_DIR / fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  guardada: {fname}")

# -- Instrucciones por iteración (una gráfica por tipo) -----------------------
for tipo, grupo_tipo in raw.groupby("tipo"):
    fig, ax = plt.subplots(figsize=(8, 4))
    for threads, grupo_th in grupo_tipo.groupby("threads"):
        grupo_th = grupo_th.sort_values("run")
        ax.plot(grupo_th["run"], grupo_th["instructions"],
                marker="o", markersize=4, linewidth=1.5,
                label=f"{threads} hilo{'s' if threads != 1 else ''}")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Instrucciones")
    ax.set_title(f"Variación de instrucciones por iteración – {tipo.upper()}")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.5)
    fig.tight_layout()
    fname = f"instrucciones_{tipo}.png"
    fig.savefig(OUT_DIR / fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  guardada: {fname}")

# -- Cache misses por iteración (una gráfica por tipo) ------------------------
for tipo, grupo_tipo in raw.groupby("tipo"):
    fig, ax = plt.subplots(figsize=(8, 4))
    for threads, grupo_th in grupo_tipo.groupby("threads"):
        grupo_th = grupo_th.sort_values("run")
        ax.plot(grupo_th["run"], grupo_th["cache-misses"],
                marker="o", markersize=4, linewidth=1.5,
                label=f"{threads} hilo{'s' if threads != 1 else ''}")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Cache misses")
    ax.set_title(f"Variación de cache misses por iteración – {tipo.upper()}")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.5)
    fig.tight_layout()
    fname = f"cachemisses_{tipo}.png"
    fig.savefig(OUT_DIR / fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  guardada: {fname}")

