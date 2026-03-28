/* Main del coarse grained:
    - multiplicaciones livianas: NO cambian de hilo
    - multiplicaciones pesadas: pueden gatillar stall + context switch
    - stalls: probabilidad por tipo de operación (OpType), deterministas por evento (tag,j,k)
    - context switch: solo ocurre cuando hay stall (si hay otro hilo disponible)
*/
#include "algorythmCGMT.cpp"
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

static bool tryLoadDatasetPath(const std::filesystem::path& p,
                               std::vector<std::vector<double>>& X,
                               std::vector<double>& Y) {
    if (!std::filesystem::exists(p)) return false;
    return loadDataset(p.string(), X, Y) && !X.empty();
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
    namespace fs = std::filesystem;
    std::vector<fs::path> candidates;
    try {
        fs::path exe = (argc > 0) ? fs::canonical(argv[0]) : fs::path{};
        fs::path exeDir = exe.has_parent_path() ? exe.parent_path() : fs::current_path();
        candidates.push_back(exeDir / "../../dataset.txt");
        candidates.push_back(exeDir / "../../../dataset.txt");
    } catch (...) {
        // si canonical() falla, igual probamos con cwd
    }
    candidates.push_back(fs::current_path() / "dataset.txt");
    candidates.push_back(fs::current_path() / "../../dataset.txt");
    candidates.push_back(fs::current_path() / "../../../dataset.txt");

    bool loaded = false;
    fs::path loadedPath;
    for (const auto& p : candidates) {
        if (tryLoadDatasetPath(p, X_all, Y_all)) {
            loaded = true;
            loadedPath = p;
            break;
        }
    }
    if (!loaded) {
        return 1;
    }

    // 80/20 train/test
    int n_train = (int)(X_all.size() * 0.8);
    vector<vector<double>> X_train(X_all.begin(), X_all.begin() + n_train);
    vector<double> Y_train(Y_all.begin(), Y_all.begin() + n_train);
    vector<vector<double>> X_test(X_all.begin() + n_train, X_all.end());
    vector<double> Y_test(Y_all.begin() + n_train, Y_all.end());

    CoarseGrainedScheduler scheduler(num_threads, 1, 1, seed);
    NeuralNetworkCoarseGrained nn(&scheduler, seed);

    auto start = high_resolution_clock::now();

    nn.train(X_train, Y_train, EPOCHS, LEARNING_RATE);

    auto end = high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    // (silencioso) evaluamos para mantener el mismo flujo de trabajo/ejecución
    (void)nn.evaluate(X_test, Y_test);

    // CSV metrics
    append_metrics_csv(
        "cgmt",
        num_threads,
        INPUT_DIM,
        HIDDEN_NEURONS,
        OUTPUT_DIM,
        EPOCHS,
        (int)X_all.size(),
        /*total_mult_en_ciclos*/ scheduler.get_light_count() + scheduler.get_heavy_count(),
        /*nops_stalls*/ scheduler.get_stall_count(),
        /*ciclos_simulados*/ scheduler.get_global_clock(),
        /*context_switches*/ scheduler.get_context_switch_count(),
        /*tiempo_ejec_s*/ duration.count());

    return 0;
}
