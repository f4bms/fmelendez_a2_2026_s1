#include "algorythmTh.cpp"

#include <chrono>
#include <iomanip>

#include "../common/runner.h"

int main(int argc, char* argv[]) {
    int num_threads = 4;
    if (argc >= 2) {
        num_threads = max(1, atoi(argv[1]));
    }

    cout << "-----------------------------------------------" << endl;
    cout << " Ejecución Multi-Thread" << endl;
    cout << "  Threads: " << num_threads << endl;
    cout << "-----------------------------------------------" << endl;

    vector<vector<double>> X_all;
    vector<double> Y_all;
    if (!loadDataset("../dataset.txt", X_all, Y_all) || X_all.empty()) {
        cerr << "Error: el dataset está vacío o no encontrado." << endl;
        return 1;
    }

    auto split = split_train_test(X_all, Y_all, 0.8);
    auto& X_train = split.X_train;
    auto& Y_train = split.Y_train;
    auto& X_test  = split.X_test;
    auto& Y_test  = split.Y_test;

    cout << "Entrenamiento: " << X_train.size() << " muestras" << endl;
    cout << "Pruebas:       " << X_test.size() << " muestras\n" << endl;

    NeuralNetworkThreaded nn(num_threads);

    auto start = std::chrono::high_resolution_clock::now();

    cout << "Se inicia entrenamiento" << endl;
    cout << "----------------------------------------" << endl;

    nn.train(X_train, Y_train, EPOCHS, LEARNING_RATE);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    cout << "\nEvaluación del modelo:" << endl;
    double test_mse = nn.evaluate(X_test, Y_test);
    runner::print_metrics(test_mse);
    runner::print_predictions_expected_basefunction(nn, X_test, 5);

    cout << "\nTiempo de ejecución: " << fixed << setprecision(3) << duration.count() << " s" << endl;
    cout << "-----------------------------------------------" << endl;

    return 0;
}
