#include "algorythm.cpp"

#include <chrono>
#include <iomanip>

int main() {
    cout << "-----------------------------------------------" << endl;
    cout << " Ejecución \"Normal\" " << endl;
    cout << "  Threads: 0 " << endl;
    cout << "-----------------------------------------------" << endl;
    
    // Cargar todos los datos
    vector<vector<double>> X_all;
    vector<double> Y_all;
    loadDataset("../dataset.txt", X_all, Y_all);
    
    int n_samples = X_all.size();
    int n_train = (int)(n_samples * 0.8);
    
    vector<vector<double>> X_train(X_all.begin(), X_all.begin() + n_train);
    vector<vector<double>> X_test(X_all.begin() + n_train, X_all.end());
    vector<double> Y_train(Y_all.begin(), Y_all.begin() + n_train);
    vector<double> Y_test(Y_all.begin() + n_train, Y_all.end());
    
    X_all.clear();
    Y_all.clear();
    
    cout << "Entrenamiento: " << X_train.size() << endl;
    cout << "Pruebas: " << X_test.size() << endl;
    cout << "----------------------------------------" << endl;
    cout << endl;
    
    // Crear red neuronal
    NeuralNetwork nn;

    auto start = std::chrono::high_resolution_clock::now();
    
    cout << "Se inicia entrenamiento" << endl;
    cout << "----------------------------------------" << endl;

    nn.train(X_train, Y_train, EPOCHS, LEARNING_RATE);
    
    cout << "Evaluacion del modelo:" << endl;
    cout << endl;
    
    double test_mse = nn.evaluate(X_test, Y_test);
    cout << "MSE: " << test_mse << endl;
    cout << "RMSE: " << sqrt(test_mse) << endl;
    cout << endl;
    
    cout << "Muestras de prediccion:" << endl;
    
    for (int i = 0; i < 5 && i < X_test.size(); i++) {
        double input[INPUT_DIM];
        for (int j = 0; j < INPUT_DIM; j++) {
            input[j] = X_test[i][j];
        }
        
        double prediction = nn.predict(input);
        double real = Y_test[i];
        double error = abs(prediction - real);

        cout << "Input: [" << input[0] << ", " << input[1] << ", " << input[2] 
             << "] -> Predicción: " << prediction << " | Esperado: " << real << endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    cout << "-------------------------------------------------" << endl;
    cout << "\nTiempo de ejecución: " << fixed << setprecision(3) << duration.count() << " s" << endl;
    cout << "-------------------------------------------------" << endl;

    return 0;
}
