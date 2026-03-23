/* Main del coarse grained:
    - multiplicaciones livianas: NO cambian de hilo
    - multiplicaciones pesadas: pueden gatillar stall + context switch
    - el switch NO es determinístico: ocurre cada N pesadas, donde N es aleatorio (thread-safe)
*/
#include "algorythmCGMT.cpp"

int main(int argc, char* argv[]) {
    int num_threads = 2;

    cout << "-----------------------------------------------" << endl;
    cout << " Ejecución de Coarse Grained " << endl;
    cout << "  Threads: " << num_threads << endl;
    cout << "  Stall por heavy (cada N pesadas aleatorio) | latencia: 1 ciclo | cs penalty: 3 ciclos" << endl;
    cout << "  N heavy-switch aleatorio en rango: [2, 8]" << endl;
    cout << "  Omitir context switch en ~15% de stalls (random, thread-safe)" << endl;
    cout << "-----------------------------------------------" << endl;

    // Cargar dataset
    vector<vector<double>> X_all;
    vector<double> Y_all;
    loadDataset("../dataset.txt", X_all, Y_all);

    // 80/20 train/test
    int n_train = (int)(X_all.size() * 0.8);
    vector<vector<double>> X_train(X_all.begin(), X_all.begin() + n_train);
    vector<double> Y_train(Y_all.begin(), Y_all.begin() + n_train);
    vector<vector<double>> X_test(X_all.begin() + n_train, X_all.end());
    vector<double> Y_test(Y_all.begin() + n_train, Y_all.end());

    cout << "Entrenamiento: " << X_train.size() << " muestras" << endl;
    cout << "Pruebas: " << X_test.size() << " muestras\n" << endl;

    CoarseGrainedScheduler scheduler(num_threads, 1, 3);
    scheduler.set_heavy_switch_random_range(2, 8);
    scheduler.set_skip_context_switch_chance_percent(15);
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
