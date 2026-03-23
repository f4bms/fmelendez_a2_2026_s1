// Algoritmo multithread (asignación dinámica de neuronas).
// Similar a `algorythm.cpp`, pero paraleliza por neuronas usando una cola/índice atómico.
//
// Idea:
// - En forward:
//   * Cada neurona oculta j se puede calcular en paralelo.
//   * Luego, cada neurona de salida (aquí OUTPUT_DIM=1) reduce sobre hidden_output.
// - En backward:
//   * hidden_delta[j] se puede calcular en paralelo.
//   * Update de wo[i][0] (por i) y wh[*][j] (por j) también se puede paralelizar.
//
// Importante:
// - Este archivo NO simula ciclos/scheduler como FGMT/CGMT.
// - Solo busca paralelizar el cómputo manteniendo la matemática del algoritmo base.

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>

using namespace std;

// Constantes de la red neuronal
const int INPUT_DIM = 3;
const int HIDDEN_NEURONS = 30;
const int OUTPUT_DIM = 1;
const double LEARNING_RATE = 0.08;
const int EPOCHS = 1500;

static inline double basefunction(double x[], int d) {
    double sum = 0;
    for (int i = 0; i < d; i++) {
        sum += sin(x[i]) + 0.3 * x[i] * x[i];
    }
    return sum;
}

static inline double tanh_activation(double x) {
    return tanh(x);
}

static inline double tanh_derivative(double x) {
    double t = tanh(x);
    return 1.0 - t * t;
}

class NeuralNetworkThreaded {
private:
    double wh[INPUT_DIM][HIDDEN_NEURONS];
    double bh[HIDDEN_NEURONS];
    double wo[HIDDEN_NEURONS][OUTPUT_DIM];
    double bo[OUTPUT_DIM];

    double hidden_input[HIDDEN_NEURONS];
    double hidden_output[HIDDEN_NEURONS];
    double output_input[OUTPUT_DIM];
    double output[OUTPUT_DIM];

    int num_threads;

    template <typename Fn>
    void parallel_for_dynamic(int total_items, Fn fn) {
        atomic<int> next{0};
        vector<thread> threads;
        threads.reserve(num_threads);

        for (int t = 0; t < num_threads; t++) {
            threads.emplace_back([&, t] {
                while (true) {
                    int idx = next.fetch_add(1, memory_order_relaxed);
                    if (idx >= total_items) break;
                    fn(idx, t);
                }
            });
        }
        for (auto &th : threads) th.join();
    }

public:
    explicit NeuralNetworkThreaded(int n_threads)
        : num_threads(max(1, n_threads)) {
        srand(time(NULL));

        for (int i = 0; i < INPUT_DIM; i++) {
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                wh[i][j] = ((double)rand() / RAND_MAX - 0.5) * sqrt(2.0 / INPUT_DIM);
            }
        }

        for (int j = 0; j < HIDDEN_NEURONS; j++) bh[j] = 0.0;

        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            for (int j = 0; j < OUTPUT_DIM; j++) {
                wo[i][j] = ((double)rand() / RAND_MAX - 0.5) * sqrt(2.0 / HIDDEN_NEURONS);
            }
        }

        for (int j = 0; j < OUTPUT_DIM; j++) bo[j] = 0.0;
    }

    double forward(const double input[]) {
        // Capa oculta: cada neurona j es independiente.
        parallel_for_dynamic(HIDDEN_NEURONS, [&](int j, int /*tid*/) {
            double sum = bh[j];
            for (int i = 0; i < INPUT_DIM; i++) {
                sum += input[i] * wh[i][j];
            }
            hidden_input[j] = sum;
            hidden_output[j] = tanh_activation(sum);
        });

        // Capa de salida: OUTPUT_DIM=1. Reducimos en un solo hilo (simple y seguro).
        for (int outj = 0; outj < OUTPUT_DIM; outj++) {
            double sum = bo[outj];
            for (int i = 0; i < HIDDEN_NEURONS; i++) {
                sum += hidden_output[i] * wo[i][outj];
            }
            output_input[outj] = sum;
            output[outj] = sum;
        }

        return output[0];
    }

    void backward(const double input[], double target, double learning_rate) {
        double output_delta[OUTPUT_DIM];
        for (int j = 0; j < OUTPUT_DIM; j++) {
            output_delta[j] = output[j] - target;
        }

        // hidden_delta[j] independiente por neurona.
        double hidden_delta[HIDDEN_NEURONS];
        parallel_for_dynamic(HIDDEN_NEURONS, [&](int j, int /*tid*/) {
            double hidden_error = 0.0;
            for (int k = 0; k < OUTPUT_DIM; k++) {
                hidden_error += output_delta[k] * wo[j][k];
            }
            hidden_delta[j] = hidden_error * tanh_derivative(hidden_input[j]);
        });

        // Actualizar wo por i (independiente por fila i porque OUTPUT_DIM=1).
        parallel_for_dynamic(HIDDEN_NEURONS, [&](int i, int /*tid*/) {
            for (int j = 0; j < OUTPUT_DIM; j++) {
                wo[i][j] -= learning_rate * output_delta[j] * hidden_output[i];
            }
        });

        // Bias output (OUTPUT_DIM=1): secuencial.
        for (int j = 0; j < OUTPUT_DIM; j++) {
            bo[j] -= learning_rate * output_delta[j];
        }

        // Actualizar wh por neurona oculta j (cada j actualiza wh[*][j]).
        parallel_for_dynamic(HIDDEN_NEURONS, [&](int j, int /*tid*/) {
            for (int i = 0; i < INPUT_DIM; i++) {
                wh[i][j] -= learning_rate * hidden_delta[j] * input[i];
            }
        });

        // Bias hidden: paralelo por neurona j.
        parallel_for_dynamic(HIDDEN_NEURONS, [&](int j, int /*tid*/) {
            bh[j] -= learning_rate * hidden_delta[j];
        });
    }

    void train(const vector<vector<double>>& X, const vector<double>& Y, int epochs, double learning_rate) {
        int n_samples = (int)X.size();
        for (int epoch = 0; epoch < epochs; epoch++) {
            double total_loss = 0.0;

            for (int i = 0; i < n_samples; i++) {
                double in[INPUT_DIM];
                for (int j = 0; j < INPUT_DIM; j++) in[j] = X[i][j];

                double prediction = forward(in);
                double error = prediction - Y[i];
                total_loss += error * error;

                backward(in, Y[i], learning_rate);
            }

            if ((epoch + 1) % 100 == 0) {
                double mse = total_loss / n_samples;
                cout << "Epoch " << (epoch + 1) << "/" << epochs << " - MSE: " << mse << endl;
            }
        }
    }

    double predict(const double input[]) { return forward(input); }

    double evaluate(const vector<vector<double>>& X, const vector<double>& Y) {
        double total_loss = 0.0;
        int n_samples = (int)X.size();
        for (int i = 0; i < n_samples; i++) {
            double in[INPUT_DIM];
            for (int j = 0; j < INPUT_DIM; j++) in[j] = X[i][j];

            double prediction = forward(in);
            double error = prediction - Y[i];
            total_loss += error * error;
        }
        return total_loss / n_samples;
    }

    int get_num_threads() const { return num_threads; }
};

void loadDataset(const string& filename, vector<vector<double>>& X, vector<double>& Y) {
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error: No se pudo abrir el archivo de datos" << endl;
        return;
    }

    double x1, x2, x3, y;
    while (file >> x1 >> x2 >> x3 >> y) {
        X.push_back({x1, x2, x3});
        Y.push_back(y);
    }

    file.close();
    cout << "Dataset cargado: " << X.size() << " muestras" << endl;
}
