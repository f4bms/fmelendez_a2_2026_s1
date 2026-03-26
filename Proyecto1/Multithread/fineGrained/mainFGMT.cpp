/* Main del fine grained, se hace por cada context switch por cada multiplicación,
   con soporte para multiplicaciones light (forward), heavy (backward) y fuera de ciclos.
   El stall se simula cada 10 multiplicaciones por hilo → NOP + cede turno. */
#include "algorythmFGMT.cpp"

#include <filesystem>
#include <fstream>

static void append_metrics_csv(const std::string& tipo,
                               int cant_threads,
                               int input_dim,
                               int hidden_neurons,
                               int output_dim,
                               int epochs,
                               int dataset_elements,
                               long long total_mult_en_ciclos,
                               long long nops_stalls,
                               long long ciclos_simulados,
                               long long context_switches,
                               double tiempo_ejec_s) {
    namespace fs = std::filesystem;
    const fs::path outPath = fs::current_path() / "metrics.csv";

    const bool needHeader = (!fs::exists(outPath)) || (fs::file_size(outPath) == 0);
    std::ofstream out(outPath, std::ios::app);
    if (!out) return;

    if (needHeader) {
        out << "tipo,cant_threads,input_dim,hidden_neurons,output_dim,epochs,dataset_elements,total_mult_en_ciclos,nops_stalls,ciclos_simulados,context_switches,tiempo_ejecucion_s\n";
    }

    out << tipo << ','
        << cant_threads << ','
        << input_dim << ','
        << hidden_neurons << ','
        << output_dim << ','
        << epochs << ','
        << dataset_elements << ','
        << total_mult_en_ciclos << ','
        << nops_stalls << ','
        << ciclos_simulados << ','
        << context_switches << ','
        << std::fixed << std::setprecision(6) << tiempo_ejec_s
        << "\n";
}

static bool tryLoadDataset(const std::string& path,
                           std::vector<std::vector<double>>& X,
                           std::vector<double>& Y) {
    X.clear();
    Y.clear();
    loadDataset(path, X, Y);
    return !X.empty();
}

int main(int argc, char* argv[]) {
    int num_threads = 2;
    uint32_t seed = 42u;
    if (argc >= 2) {
        try {
            num_threads = std::max(1, std::stoi(argv[1]));
        } catch (...) {
            return 1;
        }
    }
    if (argc >= 3) {
        try {
            seed = static_cast<uint32_t>(std::stoul(argv[2]));
        } catch (...) {
            return 1;
        }
    }

    // Cargar dataset
    vector<vector<double>> X_all;
    vector<double> Y_all;

    // Ruta robusta: intenta resolver relativo al ejecutable y también algunos fallbacks.
    namespace fs = std::filesystem;
    fs::path exeDir = (argc > 0 && argv[0]) ? fs::absolute(fs::path(argv[0])).parent_path() : fs::current_path();

    std::vector<fs::path> candidates = {
        exeDir / "../../dataset.txt",                  // ejecución desde fineGrained/
        exeDir / "../dataset.txt",                     // ejecución desde Multithread/
        exeDir / "dataset.txt",                        // ejecución con dataset junto al binario
        fs::current_path() / "dataset.txt",            // cwd tiene dataset
        fs::current_path() / "Proyecto1/dataset.txt",  // ejecución desde raíz del repo
        fs::current_path() / "../dataset.txt",         // ejecución desde ./Multithread/fineGrained
    };

    bool loaded = false;
    for (const auto& p : candidates) {
        if (tryLoadDataset(p.lexically_normal().string(), X_all, Y_all)) {
            loaded = true;
            break;
        }
    }

    if (!loaded) {
        return 1;
    }

    // 80% entrenamiento, 20% pruebas
    int n_train = (int)(X_all.size() * 0.8);
    vector<vector<double>> X_train(X_all.begin(), X_all.begin() + n_train);
    vector<double> Y_train(Y_all.begin(), Y_all.begin() + n_train);
    vector<vector<double>> X_test(X_all.begin() + n_train, X_all.end());
    vector<double> Y_test(Y_all.begin() + n_train, Y_all.end());
    
    // Crear scheduler y red neuronal
    RoundRobinScheduler scheduler(num_threads, 1, seed);
    NeuralNetworkFineGrained nn(&scheduler, seed);
    
    auto start = high_resolution_clock::now();
    
    // Entrenar
    nn.train(X_train, Y_train, EPOCHS, LEARNING_RATE);
    
    auto end = high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    
    // (silencioso) evaluamos para mantener el mismo flujo de trabajo/ejecución
    (void)nn.evaluate(X_test, Y_test);

    // CSV metrics
    append_metrics_csv(
        "fgmt",
        num_threads,
        INPUT_DIM,
        HIDDEN_NEURONS,
        OUTPUT_DIM,
        EPOCHS,
        (int)X_all.size(),
        /*total_mult_en_ciclos*/ scheduler.get_light_count() + scheduler.get_heavy_count(),
        /*nops_stalls*/ scheduler.get_stall_nop_count(),
        /*ciclos_simulados*/ scheduler.get_global_clock(),
        /*context_switches*/ scheduler.get_context_switch_count(),
        /*tiempo_ejec_s*/ duration.count());
    
    return 0;
}