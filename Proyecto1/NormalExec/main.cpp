#include "algorythm.cpp"

#include <chrono>
#include <iomanip>

#include "../common/runner.h"

int main() {
    cout << "-----------------------------------------------" << endl;
    cout << " Ejecución \"Normal\" " << endl;
    cout << "  Threads: 0 " << endl;
    cout << "-----------------------------------------------" << endl;
    
    // Cargar todos los datos
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
    
    cout << "Entrenamiento: " << X_train.size() << endl;
    cout << "Pruebas: " << X_test.size() << endl;
    cout << "----------------------------------------" << endl;
    cout << endl;
    
    NeuralNetwork nn;

    auto start = std::chrono::high_resolution_clock::now();
    
    cout << "Se inicia entrenamiento" << endl;
    cout << "----------------------------------------" << endl;

    nn.train(X_train, Y_train, EPOCHS, LEARNING_RATE);
    
    cout << "Evaluacion del modelo:" << endl;
    cout << endl;
    
    double test_mse = nn.evaluate(X_test, Y_test);
    runner::print_metrics(test_mse);
    runner::print_predictions_expected_y(nn, X_test, Y_test, 5);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    cout << "-------------------------------------------------" << endl;
    cout << "\nTiempo de ejecucion: " << fixed << setprecision(3) << duration.count() << " s" << endl;
    cout << "-------------------------------------------------" << endl;

    return 0;
}
