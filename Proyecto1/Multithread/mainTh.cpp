#include "algorythmTh.cpp"

#include <chrono>
#include <iomanip>

int main(int argc, char* argv[]) {
    int num_threads = 4;
    if (argc >= 2) {
        num_threads = max(1, atoi(argv[1]));
    }

    cout << "-----------------------------------------------" << endl;
    cout << " Ejecución de Multi-Thread dinámico (por neurona)" << endl;
    cout << "  Threads: " << num_threads << endl;
    cout << "-----------------------------------------------" << endl;

    vector<vector<double>> X_all;
    vector<double> Y_all;
    loadDataset("../dataset.txt", X_all, Y_all);

    if (X_all.empty()) {
        cerr << "Error: dataset vacío o no encontrado." << endl;
        return 1;
    }

    int n_train = (int)(X_all.size() * 0.8);
    vector<vector<double>> X_train(X_all.begin(), X_all.begin() + n_train);
    vector<double> Y_train(Y_all.begin(), Y_all.begin() + n_train);
    vector<vector<double>> X_test(X_all.begin() + n_train, X_all.end());
    vector<double> Y_test(Y_all.begin() + n_train, Y_all.end());

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
    cout << "MSE:  " << fixed << setprecision(6) << test_mse << endl;
    cout << "RMSE: " << sqrt(test_mse) << endl;

    cout << "\nMuestras de predicción:" << endl;
    for (int i = 0; i < min(5, (int)X_test.size()); i++) {
        double input[INPUT_DIM];
        for (int j = 0; j < INPUT_DIM; j++) input[j] = X_test[i][j];

        double pred = nn.predict(input);
        double expected = basefunction(input, INPUT_DIM);
        cout << "Input: [" << input[0] << ", " << input[1] << ", " << input[2]
             << "] -> Predicción: " << pred
             << " | Esperado: " << expected << endl;
    }

    cout << "\nTiempo de ejecución: " << fixed << setprecision(3) << duration.count() << " s" << endl;
    cout << "-----------------------------------------------" << endl;

    return 0;
}
