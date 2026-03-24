/* Main del coarse grained:
    - multiplicaciones livianas: NO cambian de hilo
    - multiplicaciones pesadas: pueden gatillar stall + context switch
    - stalls: probabilidad por tipo de operación (OpType), deterministas por evento (tag,j,k)
    - context switch: solo ocurre cuando hay stall (si hay otro hilo disponible)
*/
#include "algorythmCGMT.cpp"
#include <filesystem>

static bool tryLoadDatasetPath(const std::filesystem::path& p,
                               std::vector<std::vector<double>>& X,
                               std::vector<double>& Y) {
    if (!std::filesystem::exists(p)) return false;
    return loadDataset(p.string(), X, Y) && !X.empty();
}

int main(int argc, char* argv[]) {
    int num_threads = 2;

    cout << "-----------------------------------------------" << endl;
    cout << " Ejecución de Coarse Grained " << endl;
    cout << "  Threads: " << num_threads << endl;
    cout << "  Stall: prob por OpType (determinista por evento tag,j,k) | penalidad stall: +1 ciclo | stall_latency extra: +1 | cs penalty: +3" << endl;
    cout << "  Scheduler: cambia de hilo SOLO cuando ocurre un stall" << endl;
    cout << "-----------------------------------------------" << endl;

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
        std::cerr << "Error: dataset vacío o no encontrado. Candidatos probados:\n";
        for (const auto& p : candidates) std::cerr << "  - " << p << "\n";
        return 1;
    }
    std::cerr << "Dataset cargado desde: " << loadedPath << "\n";

    // 80/20 train/test
    int n_train = (int)(X_all.size() * 0.8);
    vector<vector<double>> X_train(X_all.begin(), X_all.begin() + n_train);
    vector<double> Y_train(Y_all.begin(), Y_all.begin() + n_train);
    vector<vector<double>> X_test(X_all.begin() + n_train, X_all.end());
    vector<double> Y_test(Y_all.begin() + n_train, Y_all.end());

    cout << "Entrenamiento: " << X_train.size() << " muestras" << endl;
    cout << "Pruebas: " << X_test.size() << " muestras\n" << endl;

    CoarseGrainedScheduler scheduler(num_threads, 1, 3);
    NeuralNetworkCoarseGrained nn(&scheduler);

    auto start = high_resolution_clock::now();

    cout << "-------------------------" << endl;
    cout << "Se inicia el entrenmiento" << endl;
    nn.train(X_train, Y_train, EPOCHS, LEARNING_RATE);

    auto end = high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    cout << "\n Evaluación del modelo" << endl;
    double test_mse = nn.evaluate(X_test, Y_test);
    cout << "Test MSE: " << fixed << setprecision(6) << test_mse << endl;
    cout << "Test RMSE: " << sqrt(test_mse) << endl;

    cout << "\nMuestras de predicción:" << endl;
    for (int i = 0; i < min(5, (int)X_test.size()); i++) {
        double input[INPUT_DIM];
        for (int j = 0; j < INPUT_DIM; j++) input[j] = X_test[i][j];

        double pred = nn.predict(input);
        double expected = basefunction(input, INPUT_DIM);
        cout << "Input: [" << input[0] << ", " << input[1] << ", " << input[2]
             << "] -> Predicción: " << pred << " | Esperado: " << expected << endl;
    }

    scheduler.print_stats();

    cout << "Tiempo de ejecución: " << fixed << setprecision(3) << duration.count() << " s" << endl;
    cout << "---------------------\n" << endl;

    return 0;
}
