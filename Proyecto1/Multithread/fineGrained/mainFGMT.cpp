// Punto de entrada para la variante fine-grained del simulador.
// El scheduler cambia de hilo después de CADA multiplicación (round-robin),
// lo que produce el mayor overhead de context switch entre las tres variantes
// (normal, coarse-grained, fine-grained).
// Los stalls por cache miss se simulan con un hash determinista:
//   DOT_PRODUCT → 2.4% de probabilidad, 8 ciclos de penalización
//   ACTIVATION  → 0.6% de probabilidad, 3 ciclos de penalización
#include "algorythmFGMT.cpp"

#include <filesystem>
#include <fstream>

// El CSV acumula resultados de todas las variantes (normal, cgmt, fgmt)
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

// Intenta cargar el dataset desde una ruta dada.
// Retorna true si se cargaron datos; false si el archivo no existe o está vacío.
static bool tryLoadDataset(const std::string& path, std::vector<std::vector<double>>& X, std::vector<double>& Y) {
    X.clear();
    Y.clear();
    loadDataset(path, X, Y);
    return !X.empty();
}

int main(int argc, char* argv[]) {
    // Parámetros con valores por defecto: 2 hilos, seed 42.
    // La seed controla tanto la inicialización de pesos como los stalls simulados,
    // garantizando resultados reproducibles para el mismo par (threads, seed).
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

    vector<vector<double>> X_all;
    vector<double> Y_all;

    //se crea para evitar problemas con las direcciones
    namespace fs = std::filesystem;
    fs::path exeDir = (argc > 0 && argv[0]) ? fs::absolute(fs::path(argv[0])).parent_path() : fs::current_path();

    std::vector<fs::path> candidates = {
        exeDir / "../../dataset.txt",
        exeDir / "../dataset.txt",
        exeDir / "dataset.txt",
        fs::current_path() / "dataset.txt",
        fs::current_path() / "Proyecto1/dataset.txt",
        fs::current_path() / "../dataset.txt",
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

    // Split 80/20: los primeros n_train registros son entrenamiento,
    // el resto prueba. El orden del dataset determina la partición.
    int n_train = (int)(X_all.size() * 0.8);
    vector<vector<double>> X_train(X_all.begin(), X_all.begin() + n_train);
    vector<double> Y_train(Y_all.begin(), Y_all.begin() + n_train);
    vector<vector<double>> X_test(X_all.begin() + n_train, X_all.end());
    vector<double> Y_test(Y_all.begin() + n_train, Y_all.end());

    // El scheduler se crea con context_switch_penalty = 1 ciclo base.
    // La red usa la misma seed para que los pesos iniciales sean comparables
    // entre variantes cuando se usa la misma seed desde línea de comandos.
    RoundRobinScheduler scheduler(num_threads, 1, seed);
    NeuralNetworkFineGrained nn(&scheduler, seed);

    auto start = high_resolution_clock::now();
    nn.train(X_train, Y_train, EPOCHS, LEARNING_RATE);
    auto end = high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    (void)nn.evaluate(X_test, Y_test);

    append_metrics_csv(
        "fgmt",
        num_threads,
        INPUT_DIM,
        HIDDEN_NEURONS,
        OUTPUT_DIM,
        EPOCHS,
        (int)X_all.size(),
        /*total_mult_en_ciclos*/ scheduler.get_light_count() + scheduler.get_heavy_count(),
        /*nops_stalls*/          scheduler.get_stall_nop_count(),
        /*ciclos_simulados*/     scheduler.get_global_clock(),
        /*context_switches*/     scheduler.get_context_switch_count(),
        /*tiempo_ejec_s*/        duration.count());

    return 0;
}
