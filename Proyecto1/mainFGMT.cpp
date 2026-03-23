/* Main del fine grained, se hace por cada context switch por cada multiplicación,
   con soporte para multiplicaciones light (forward), heavy (backward) y fuera de ciclos.
   El stall se simula cada 10 multiplicaciones por hilo → NOP + cede turno. */
#include "algorythmFGMT.cpp"

int main(int argc, char* argv[]) {
    int num_threads = 2;
    
    cout << "-----------------------------------------------" << endl;
    cout << " Ejecución de Fine Grained Multi-Thread" << endl;
    cout << "  Threads:        " << num_threads << endl;
    cout << "  Stall cada:     " << 10 << " multiplicaciones por hilo" << endl;
    cout << "  Light muls:     forward pass (input->hidden, hidden->output)" << endl;
    cout << "  Heavy muls:     backward pass (gradientes y deltas)" << endl;
    cout << "  Fuera ciclos:   actualizaciones de pesos (lr * grad)" << endl;
    cout << "-----------------------------------------------" << endl;

    // Cargar dataset
    vector<vector<double>> X_all;
    vector<double> Y_all;
    loadDataset("dataset.txt", X_all, Y_all);

    if (X_all.empty()) {
        cerr << "Error: dataset vacío o no encontrado." << endl;
        return 1;
    }

    // 80% entrenamiento, 20% pruebas
    int n_train = (int)(X_all.size() * 0.8);
    vector<vector<double>> X_train(X_all.begin(), X_all.begin() + n_train);
    vector<double> Y_train(Y_all.begin(), Y_all.begin() + n_train);
    vector<vector<double>> X_test(X_all.begin() + n_train, X_all.end());
    vector<double> Y_test(Y_all.begin() + n_train, Y_all.end());
    
    cout << "Entrenamiento: " << X_train.size() << " muestras" << endl;
    cout << "Pruebas:       " << X_test.size()  << " muestras\n" << endl;

    // Crear scheduler y red neuronal
    RoundRobinScheduler scheduler(num_threads, 1);
    NeuralNetworkFineGrained nn(&scheduler);
    
    auto start = high_resolution_clock::now();
    
    // Entrenar
    cout << "-------------------------" << endl;
    cout << "Se inicia el entrenamiento" << endl;
    nn.train(X_train, Y_train, EPOCHS, LEARNING_RATE);
    
    auto end = high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    
    // Evaluación
    cout << "\n Evaluación del modelo" << endl;
    double test_mse = nn.evaluate(X_test, Y_test);
    cout << "Test MSE:  " << fixed << setprecision(6) << test_mse << endl;
    cout << "Test RMSE: " << sqrt(test_mse) << endl;
    
    // Algunas predicciones para verificar funcionamiento
    cout << "\nMuestras de predicción:" << endl;
    for (int i = 0; i < min(5, (int)X_test.size()); i++) {
        double input[INPUT_DIM];
        for (int j = 0; j < INPUT_DIM; j++) {
            input[j] = X_test[i][j];
        }
        double pred     = nn.predict(input);
        double expected = basefunction(input, INPUT_DIM);
        cout << "Input: [" << input[0] << ", " << input[1] << ", " << input[2]
             << "] -> Predicción: " << pred
             << " | Esperado: "    << expected << endl;
    }
    
    // Estadísticas del scheduler (light, heavy, NOPs, clock, fuera de ciclos)
    scheduler.print_stats();
    
    // Métricas adicionales calculadas
    long long total_ciclos = scheduler.get_light_count() + scheduler.get_heavy_count();
    if (total_ciclos > 0) {
        double pct_light = 100.0 * scheduler.get_light_count() / total_ciclos;
        double pct_heavy = 100.0 * scheduler.get_heavy_count() / total_ciclos;
        double nop_ratio = 100.0 * scheduler.get_stall_nop_count() / scheduler.get_global_clock();
        cout << " --- distribución de ciclos ---" << endl;
        cout << "  Light: " << fixed << setprecision(1) << pct_light << "%" << endl;
        cout << "  Heavy: " << pct_heavy << "%" << endl;
        cout << "  NOPs / total clock: " << nop_ratio << "%" << endl;
        cout << "------------------------------\n" << endl;
    }

    cout << "Tiempo de ejecución: " << fixed << setprecision(3) << duration.count() << " s" << endl;
    cout << "---------------------\n" << endl;
    
    return 0;
}