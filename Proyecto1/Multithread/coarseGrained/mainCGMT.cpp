/* Main del coarse grained:
    - multiplicaciones livianas: NO cambian de hilo
    - multiplicaciones pesadas: pueden gatillar stall + context switch
    - el switch NO es determinístico: ocurre cada N pesadas, donde N es aleatorio (thread-safe)
*/
#include "algorythmCGMT.cpp"

#include "../../common/runner.h"

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
    if (!loadDataset("../dataset.txt", X_all, Y_all) || X_all.empty()) {
        cerr << "Error: dataset vacío o no encontrado." << endl;
        return 1;
    }

    auto split = split_train_test(X_all, Y_all, 0.8);
    auto& X_train = split.X_train;
    auto& Y_train = split.Y_train;
    auto& X_test  = split.X_test;
    auto& Y_test  = split.Y_test;

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
    runner::print_metrics(test_mse);
    runner::print_predictions_expected_basefunction(nn, X_test, 5);

    scheduler.print_stats();

    cout << "Tiempo de ejecución: " << fixed << setprecision(3) << duration.count() << " s" << endl;
    cout << "---------------------\n" << endl;

    return 0;
}
