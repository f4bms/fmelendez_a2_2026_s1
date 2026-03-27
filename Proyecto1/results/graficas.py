import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH     = Path(__file__).parent / "resultados.csv"
METRICS_PATH = Path(__file__).parent / "metrics.csv"
OUT_DIR      = Path(__file__).parent / "graficas"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Contar N por grupo desde metrics.csv
raw = pd.read_csv(METRICS_PATH)
n_counts = raw.groupby("tipo").size()          # N por tipo (para normal)
n_par    = raw.groupby(["tipo", "cant_threads"]).size()  # N por (tipo, hilos)

n_normal = int(n_counts.get("normal", 0))
n_fgmt   = int(n_par.get(("fgmt", raw[raw["tipo"] == "fgmt"]["cant_threads"].iloc[0]), 0)) if len(raw[raw["tipo"] == "fgmt"]) else 0
n_cgmt   = int(n_par.get(("cgmt", raw[raw["tipo"] == "cgmt"]["cant_threads"].iloc[0]), 0)) if len(raw[raw["tipo"] == "cgmt"]) else 0

fgmt = df[df["tipo"] == "fgmt"].sort_values("hilos")
cgmt = df[df["tipo"] == "cgmt"].sort_values("hilos")
normal = df[df["tipo"] == "normal"].iloc[0]

# ── Ciclos de ejecución vs Hilos — FGMT ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(fgmt["hilos"], fgmt["media_aritmetica"],
        marker="o", color="#2196F3", linewidth=2, markersize=8, label="FGMT")
ax.fill_between(
    fgmt["hilos"],
    fgmt["media_aritmetica"] - fgmt["IC"],
    fgmt["media_aritmetica"] + fgmt["IC"],
    alpha=0.15, color="#2196F3", label="IC 95%"
)
ax.set_xlabel("Hilos")
ax.set_ylabel("Ciclos de ejecución (media)")
ax.set_title(f"Ciclos de Ejecución vs Hilos — FGMT (N={n_fgmt})")
ax.set_xticks(fgmt["hilos"])
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.savefig(OUT_DIR / "ciclos_fgmt.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  guardada: ciclos_fgmt.png")

# ── Ciclos de ejecución vs Hilos — CGMT ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(cgmt["hilos"], cgmt["media_aritmetica"],
        marker="s", color="#FF9800", linewidth=2, markersize=8, label="CGMT")
ax.fill_between(
    cgmt["hilos"],
    cgmt["media_aritmetica"] - cgmt["IC"],
    cgmt["media_aritmetica"] + cgmt["IC"],
    alpha=0.15, color="#FF9800", label="IC 95%"
)
ax.set_xlabel("Hilos")
ax.set_ylabel("Ciclos de ejecución (media)")
ax.set_title(f"Ciclos de Ejecución vs Hilos — CGMT (N={n_cgmt})")
ax.set_xticks(cgmt["hilos"])
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.savefig(OUT_DIR / "ciclos_cgmt.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  guardada: ciclos_cgmt.png")

# ── Ciclos de ejecución — Normal (baseline) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(4, 4))
ax.bar("Normal", normal["media_aritmetica"], color="#4CAF50", width=0.4, label="Normal")
ax.errorbar("Normal", normal["media_aritmetica"], yerr=normal["IC"],
            fmt="none", ecolor="black", capsize=6, capthick=1.5, label="IC 95%")
ax.set_ylabel("Ciclos de ejecución (media)")
ax.set_title(f"Ciclos de Ejecución — Normal (N={n_normal})")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.savefig(OUT_DIR / "ciclos_normal.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  guardada: ciclos_normal.png")

# ── Eficiencia vs Hilos — FGMT ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(fgmt["hilos"], fgmt["Eficiencia"],
        marker="o", color="#2196F3", linewidth=2, markersize=8, label="FGMT")
ax.set_xlabel("Hilos")
ax.set_ylabel("Eficiencia")
ax.set_title(f"Eficiencia vs Hilos — FGMT (N={n_fgmt})")
ax.set_xticks(fgmt["hilos"])
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.savefig(OUT_DIR / "eficiencia_fgmt.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  guardada: eficiencia_fgmt.png")

# ── Eficiencia vs Hilos — CGMT ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(cgmt["hilos"], cgmt["Eficiencia"],
        marker="s", color="#FF9800", linewidth=2, markersize=8, label="CGMT")
ax.set_xlabel("Hilos")
ax.set_ylabel("Eficiencia")
ax.set_title(f"Eficiencia vs Hilos — CGMT (N={n_cgmt})")
ax.set_xticks(cgmt["hilos"])
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.savefig(OUT_DIR / "eficiencia_cgmt.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  guardada: eficiencia_cgmt.png")

# ── SpeedUp vs Hilos — FGMT ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(fgmt["hilos"], fgmt["SpeedUp"],
        marker="o", color="#2196F3", linewidth=2, markersize=8, label="FGMT")
ax.set_xlabel("Hilos")
ax.set_ylabel("SpeedUp")
ax.set_title(f"SpeedUp vs Hilos — FGMT (N={n_fgmt})")
ax.set_xticks(fgmt["hilos"])
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.savefig(OUT_DIR / "speedup_fgmt.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  guardada: speedup_fgmt.png")

# ── SpeedUp vs Hilos — CGMT ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(cgmt["hilos"], cgmt["SpeedUp"],
        marker="s", color="#FF9800", linewidth=2, markersize=8, label="CGMT")
ax.set_xlabel("Hilos")
ax.set_ylabel("SpeedUp")
ax.set_title(f"SpeedUp vs Hilos — CGMT (N={n_cgmt})")
ax.set_xticks(cgmt["hilos"])
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.savefig(OUT_DIR / "speedup_cgmt.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  guardada: speedup_cgmt.png")

# ── SpeedUp comparado: FGMT vs CGMT ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(fgmt["hilos"], fgmt["SpeedUp"],
        marker="o", color="#2196F3", linewidth=2, markersize=8, label="FGMT")
ax.plot(cgmt["hilos"], cgmt["SpeedUp"],
        marker="s", color="#FF9800", linewidth=2, markersize=8, label="CGMT")
ax.set_xlabel("Hilos")
ax.set_ylabel("SpeedUp")
ax.set_title("SpeedUp vs Hilos — FGMT vs CGMT")
ax.set_xticks(sorted(set(fgmt["hilos"].tolist() + cgmt["hilos"].tolist())))
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.savefig(OUT_DIR / "speedup_comparado.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  guardada: speedup_comparado.png")


# ── Boxplot por tipo: un gráfico por cada tipo en metrics.csv ─────────────────
COLORES = {"fgmt": "#90CAF9", "cgmt": "#FFCC80", "normal": "#A5D6A7"}

for tipo, grupo_tipo in raw.groupby("tipo"):
    # Agrupar por cantidad de hilos dentro del tipo
    grupos_box = grupo_tipo.groupby("cant_threads")["ciclos_simulados"]
    data_box   = [g.values for _, g in grupos_box]
    labels_box = [f"{h} hilo{'s' if h != 1 else ''}" for h, _ in grupos_box]

    color = COLORES.get(tipo, "#CE93D8")
    fig, ax = plt.subplots(figsize=(max(5, len(labels_box) * 1.6), 5))
    ax.boxplot(data_box, labels=labels_box, patch_artist=True,
               boxprops=dict(facecolor=color, color="#333333"),
               medianprops=dict(color="#E53935", linewidth=2))
    ax.set_ylabel("Ciclos simulados")
    ax.set_title(f"Distribución de ciclos — {tipo.upper()}")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fname = f"boxplot_{tipo}.png"
    fig.savefig(OUT_DIR / fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  guardada: {fname}")

print(f"\nGráficas guardadas en: {OUT_DIR}/")
