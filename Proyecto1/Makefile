CXX := g++
CXXFLAGS := -O2 -std=c++17 -pthread

# Parametrización
THREADS ?= 2
RUNS ?= 1

ROOT := .
RESULTS_DIR := $(ROOT)/results

FGMT_DIR := $(ROOT)/Multithread/fineGrained
CGMT_DIR := $(ROOT)/Multithread/coarseGrained

FGMT_BIN := $(FGMT_DIR)/main_fgmt
CGMT_BIN := $(CGMT_DIR)/main_cgmt

.PHONY: all clean results_dir build run50 cgmt_build cgmt_run50

all: build

results_dir:
	@mkdir -p $(RESULTS_DIR)

build: $(FGMT_BIN)

cgmt_build: $(CGMT_BIN)

# Ejecuta una vez y guarda métricas en CSV (en cwd del binario).
.PHONY: metrics metrics_fgmt metrics_cgmt

metrics: metrics_fgmt metrics_cgmt

metrics_fgmt: build results_dir
	@mkdir -p $(RESULTS_DIR)
	@touch $(RESULTS_DIR)/metrics.csv
	@cd $(abspath $(FGMT_DIR)) && rm -f metrics.csv && ./main_fgmt $(THREADS)
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
	@cd $(abspath $(CGMT_DIR)) && rm -f metrics.csv && ./main_cgmt $(THREADS)
	@{ \
		if [ ! -s "$(RESULTS_DIR)/metrics.csv" ]; then \
			cat "$(CGMT_DIR)/metrics.csv"; \
		else \
			tail -n +2 "$(CGMT_DIR)/metrics.csv"; \
		fi; \
	} >> $(RESULTS_DIR)/metrics.csv
	@rm -f $(CGMT_DIR)/metrics.csv

.PHONY: metrics_all
metrics_all: build cgmt_build results_dir
	@$(MAKE) metrics_fgmt
	@$(MAKE) metrics_cgmt

.PHONY: metrics_runs
metrics_runs: build cgmt_build results_dir
	@for i in $$(seq 1 $(RUNS)); do \
		$(MAKE) THREADS=$(THREADS) metrics_fgmt; \
		$(MAKE) THREADS=$(THREADS) metrics_cgmt; \
	done

$(FGMT_BIN): $(FGMT_DIR)/mainFGMT.cpp $(FGMT_DIR)/algorythmFGMT.cpp $(ROOT)/common/runner.h $(ROOT)/common/nn_utils.h $(ROOT)/common/dataset.h
	$(CXX) $(CXXFLAGS) -o $@ $(FGMT_DIR)/mainFGMT.cpp

$(CGMT_BIN): $(CGMT_DIR)/mainCGMT.cpp $(CGMT_DIR)/algorythmCGMT.cpp $(ROOT)/common/nn_utils.h $(ROOT)/common/dataset.h
	$(CXX) $(CXXFLAGS) -o $@ $(CGMT_DIR)/mainCGMT.cpp


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

clean:
	@rm -f $(FGMT_BIN)
	@rm -f $(CGMT_BIN)
	@rm -rf $(RESULTS_DIR)