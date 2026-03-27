CXX := g++
CXXFLAGS := -O2 -std=c++17 -pthread

THREADS ?= 2
RUNS ?= 1
SEED ?= 42

# SMT perf — pasar SMT_CORES al invocar, ej: make perf_smt_all SMT_CORES=0,4
# stalled-cycles-backend = ciclos perdidos esperando memoria/ejecución (stalls de cache)
# stalled-cycles-frontend = ciclos perdidos en fetch/decode de instrucciones
PERF_EVENTS := task-clock,cpu_core/cycles/,cpu_core/instructions/,cpu_core/cache-misses/,cpu_core/stalled-cycles-backend/,cpu_core/stalled-cycles-frontend/

ROOT := .
RESULTS_DIR := $(ROOT)/results

FGMT_DIR := $(ROOT)/Multithread/fineGrained
CGMT_DIR := $(ROOT)/Multithread/coarseGrained

FGMT_BIN := $(FGMT_DIR)/main_fgmt
CGMT_BIN := $(CGMT_DIR)/main_cgmt
NORMAL_DIR := $(ROOT)/NormalExec
NORMAL_BIN := $(NORMAL_DIR)/main_normal

.PHONY: all clean results_dir build run50 cgmt_build cgmt_run50 normal_build resultados graficas resultados_smt

all: build

results_dir:
	@mkdir -p $(RESULTS_DIR)

build: $(FGMT_BIN)

cgmt_build: $(CGMT_BIN)

normal_build: $(NORMAL_BIN)

# Ejecuta una vez y guarda métricas en CSV (en cwd del binario).
.PHONY: metrics metrics_fgmt metrics_cgmt

metrics: metrics_fgmt metrics_cgmt metrics_normal

metrics_fgmt: build results_dir
	@mkdir -p $(RESULTS_DIR)
	@touch $(RESULTS_DIR)/metrics.csv
	@cd $(abspath $(FGMT_DIR)) && rm -f metrics.csv && ./main_fgmt $(THREADS) $(SEED)
	@{ \
		if [ ! -s "$(RESULTS_DIR)/metrics.csv" ]; then \
			cat "$(FGMT_DIR)/metrics.csv"; \
		else \
			tail -n +2 "$(FGMT_DIR)/metrics.csv"; \
		fi; \
	} >> $(RESULTS_DIR)/metrics.csv
	@rm -f $(FGMT_DIR)/metrics.csv

metrics_cgmt: cgmt_build results_dir
	@mkdir -p $(RESULTS_DIR)
	@touch $(RESULTS_DIR)/metrics.csv
	@cd $(abspath $(CGMT_DIR)) && rm -f metrics.csv && ./main_cgmt $(THREADS) $(SEED)
	@{ \
		if [ ! -s "$(RESULTS_DIR)/metrics.csv" ]; then \
			cat "$(CGMT_DIR)/metrics.csv"; \
		else \
			tail -n +2 "$(CGMT_DIR)/metrics.csv"; \
		fi; \
	} >> $(RESULTS_DIR)/metrics.csv
	@rm -f $(CGMT_DIR)/metrics.csv

metrics_normal: normal_build results_dir
	@mkdir -p $(RESULTS_DIR)
	@touch $(RESULTS_DIR)/metrics.csv
	@cd $(abspath $(NORMAL_DIR)) && rm -f metrics.csv && ./main_normal $(SEED)
	@{ \
		if [ ! -s "$(RESULTS_DIR)/metrics.csv" ]; then \
			cat "$(NORMAL_DIR)/metrics.csv"; \
		else \
			tail -n +2 "$(NORMAL_DIR)/metrics.csv"; \
		fi; \
	} >> $(RESULTS_DIR)/metrics.csv
	@rm -f $(NORMAL_DIR)/metrics.csv

.PHONY: metrics_all
metrics_all: build cgmt_build normal_build results_dir
	@$(MAKE) metrics_fgmt
	@$(MAKE) metrics_cgmt
	@$(MAKE) metrics_normal

.PHONY: metrics_runs
metrics_runs: build cgmt_build normal_build results_dir
	@for i in $$(seq 1 $(RUNS)); do \
		$(MAKE) THREADS=$(THREADS) SEED=$$i metrics_fgmt; \
		$(MAKE) THREADS=$(THREADS) SEED=$$i metrics_cgmt; \
		$(MAKE) SEED=$$i metrics_normal; \
	done

$(FGMT_BIN): $(FGMT_DIR)/mainFGMT.cpp $(FGMT_DIR)/algorythmFGMT.cpp $(ROOT)/common/runner.h $(ROOT)/common/nn_utils.h $(ROOT)/common/dataset.h
	$(CXX) $(CXXFLAGS) -o $@ $(FGMT_DIR)/mainFGMT.cpp

$(CGMT_BIN): $(CGMT_DIR)/mainCGMT.cpp $(CGMT_DIR)/algorythmCGMT.cpp $(ROOT)/common/nn_utils.h $(ROOT)/common/dataset.h
	$(CXX) $(CXXFLAGS) -o $@ $(CGMT_DIR)/mainCGMT.cpp

$(NORMAL_BIN): $(NORMAL_DIR)/main.cpp $(NORMAL_DIR)/algorythm.cpp $(ROOT)/common/runner.h $(ROOT)/common/nn_utils.h $(ROOT)/common/dataset.h
	$(CXX) $(CXXFLAGS) -o $@ $(NORMAL_DIR)/main.cpp


run50: build results_dir
	@echo "Running FGMT 50 times..."
	@rm -f $(RESULTS_DIR)/fgmt_*.txt
	@for i in $$(seq 1 50); do \
		echo "--- run $$i ---" >> $(RESULTS_DIR)/fgmt_$$i.txt; \
		cd $(abspath $(FGMT_DIR)) && ./main_fgmt >> "$(abspath $(RESULTS_DIR))/fgmt_$$i.txt"; \
		echo >> $(RESULTS_DIR)/fgmt_$$i.txt; \
	done

cgmt_run50: cgmt_build results_dir
	@echo "Running CGMT 50 times..."
	@rm -f $(RESULTS_DIR)/cgmt_*.txt
	@for i in $$(seq 1 50); do \
		echo "--- run $$i ---" >> $(RESULTS_DIR)/cgmt_$$i.txt; \
		cd $(abspath $(CGMT_DIR)) && ./main_cgmt >> "$(abspath $(RESULTS_DIR))/cgmt_$$i.txt"; \
		echo >> $(RESULTS_DIR)/cgmt_$$i.txt; \
	done

# ── SMT perf targets ──────────────────────────────────────────────────────────
MT_DIR := $(ROOT)/Multithread
MT_BIN  := $(MT_DIR)/main_th

.PHONY: mt_build perf_smt

mt_build: $(MT_BIN)

$(MT_BIN): $(MT_DIR)/mainTh.cpp $(MT_DIR)/algorythmTh.cpp $(ROOT)/common/runner.h $(ROOT)/common/nn_utils.h $(ROOT)/common/dataset.h
	$(CXX) $(CXXFLAGS) -o $@ $(MT_DIR)/mainTh.cpp

# Uso: make perf_smt SMT_CORES=0,4 THREADS=2 SEED=1 RUNS=5 LABEL=smt_par
# LABEL es opcional (default: sin_label); se guarda como primera columna "tipo"
# para identificar la corrida en el CSV sin afectar la ejecución del binario.
LABEL ?= sin_label

perf_smt: mt_build results_dir
	@[ -n "$(SMT_CORES)" ] || { echo "ERROR: define SMT_CORES, ej: make $@ SMT_CORES=0,4"; exit 1; }
	@PERF_TMP=$(abspath $(RESULTS_DIR))/.perf_tmp.csv; \
	OUT=$(abspath $(RESULTS_DIR))/smt_th.csv; \
	echo "tipo,threads,cores,run,task-clock (s),cycles,instructions,cache-misses,stalls-backend,stalls-frontend,mse" > $$OUT; \
	echo "  [warmup] precargando dataset en page cache..."; \
	cd $(abspath $(MT_DIR)) && ./main_th $(THREADS) $(SEED) $(SMT_CORES) > /dev/null 2>&1; \
	for i in $$(seq 1 $(RUNS)); do \
		MSE=$$(cd $(abspath $(MT_DIR)) && perf stat -x , \
			-e $(PERF_EVENTS) \
			-o $$PERF_TMP \
			./main_th $(THREADS) $$i $(SMT_CORES)); \
		TC=$$(grep 'task-clock'            $$PERF_TMP | cut -d',' -f1 | awk '{printf "%.6f", $$1/1000}'); \
		CY=$$(grep 'cycles'                $$PERF_TMP | cut -d',' -f1); \
		IN=$$(grep 'instructions'          $$PERF_TMP | cut -d',' -f1); \
		CM=$$(grep 'cache-misses'          $$PERF_TMP | cut -d',' -f1); \
		SB=$$(grep 'stalled-cycles-backend'  $$PERF_TMP | cut -d',' -f1); \
		SF=$$(grep 'stalled-cycles-frontend' $$PERF_TMP | cut -d',' -f1); \
		echo "$(LABEL),$(THREADS),\"($(SMT_CORES))\",$$i,$$TC,$$CY,$$IN,$$CM,$$SB,$$SF,$$MSE" >> $$OUT; \
	done; \
	rm -f $$PERF_TMP

# ── Análisis y gráficas ───────────────────────────────────────────────────────
# Uso: make graficas
# Requiere que results/metrics.csv ya esté listo con los datos deseados.
# Genera results/resultados.csv y results/graficas/*.png

PYTHON ?= python3

resultados:
	@[ -s "$(RESULTS_DIR)/metrics.csv" ] || { echo "ERROR: $(RESULTS_DIR)/metrics.csv está vacío o no existe. Corre primero 'make metrics'."; exit 1; }
	@echo "Calculando estadísticas..."
	@$(PYTHON) $(RESULTS_DIR)/metrics.py

graficas: resultados
	@echo "Generando gráficas..."
	@$(PYTHON) $(RESULTS_DIR)/graficas.py

resultados_smt:
	@[ -s "$(RESULTS_DIR)/smt_th.csv" ] || { echo "ERROR: $(RESULTS_DIR)/smt_th.csv está vacío o no existe. Corre primero 'make perf_smt'."; exit 1; }
	@echo "Calculando estadísticas SMT..."
	@$(PYTHON) $(RESULTS_DIR)/metrics_smt.py

clean:
	@rm -f $(FGMT_BIN)
	@rm -f $(CGMT_BIN)
	@rm -f $(NORMAL_BIN)
	@rm -rf $(RESULTS_DIR)