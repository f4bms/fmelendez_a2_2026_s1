/* Algoritmo de entrenamiento de una red neuronal utilizando multithreading
- Scheduling estático por bloques: cada hilo procesa un rango contiguo [start, end).
- Sin cola de tareas, igual que FGMT y CGMT.
- Inicialización de pesos determinista (mt19937 con seed fija).
*/
#include <iostream>
#include <cstdlib>
#include <vector>
#include <thread>
#include <random>
#include <algorithm>

#include "../common/nn_utils.h"
#include "../common/dataset.h"

using namespace std;

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

    // Helper: lanza hilos con scheduling estático por bloques y espera a que terminen.
    // Hilo t procesa el rango contiguo [start, end) donde start = t * block_size.
    template <typename Fn>
    void parallel_block(int total_items, Fn fn) {
        vector<thread> threads;
        threads.reserve(num_threads);
        int block_size = (total_items + num_threads - 1) / num_threads; // ceil division
        for (int t = 0; t < num_threads; t++) {
            int start = t * block_size;
            int end   = min(start + block_size, total_items);
            threads.emplace_back([&, t, start, end] {
                for (int idx = start; idx < end; idx++) {
                    fn(idx, t);
                }
            });
        }
        for (auto& th : threads) th.join();
    }

public:
    explicit NeuralNetworkThreaded(int n_threads, unsigned int fixed_seed = 42u)
        : num_threads(max(1, n_threads)) {
        // RNG determinista, igual que CGMT/FGMT
        std::mt19937 rng(fixed_seed);
        std::uniform_real_distribution<double> uni(0.0, 1.0);

        for (int i = 0; i < INPUT_DIM; i++) {
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                wh[i][j] = (uni(rng) - 0.5) * sqrt(2.0 / INPUT_DIM);
            }
        }

        for (int j = 0; j < HIDDEN_NEURONS; j++) bh[j] = 0.0;

        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            for (int j = 0; j < OUTPUT_DIM; j++) {
                wo[i][j] = (uni(rng) - 0.5) * sqrt(2.0 / HIDDEN_NEURONS);
            }
        }

        for (int j = 0; j < OUTPUT_DIM; j++) bo[j] = 0.0;
    }

    double forward(const double input[]) {
        // Capa oculta: cada neurona j es independiente → strided por hilo.
        parallel_block(HIDDEN_NEURONS, [&](int j, int /*tid*/) {
            double sum = bh[j];
            for (int i = 0; i < INPUT_DIM; i++) {
                sum += input[i] * wh[i][j];
            }
            hidden_input[j] = sum;
            hidden_output[j] = tanh_activation(sum);
        });

        // Capa de salida: OUTPUT_DIM=1, secuencial (sin races).
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

        // hidden_delta[j]: independiente por neurona → strided.
        double hidden_delta[HIDDEN_NEURONS];
        parallel_block(HIDDEN_NEURONS, [&](int j, int /*tid*/) {
            double hidden_error = 0.0;
            for (int k = 0; k < OUTPUT_DIM; k++) {
                hidden_error += output_delta[k] * wo[j][k];
            }
            double t = tanh_activation(hidden_input[j]);
            hidden_delta[j] = hidden_error * (1.0 - t * t);
        });

        // Actualizar wo: independiente por fila i (OUTPUT_DIM=1).
        parallel_block(HIDDEN_NEURONS, [&](int i, int /*tid*/) {
            for (int j = 0; j < OUTPUT_DIM; j++) {
                wo[i][j] -= learning_rate * output_delta[j] * hidden_output[i];
            }
        });

        // Bias output: secuencial.
        for (int j = 0; j < OUTPUT_DIM; j++) {
            bo[j] -= learning_rate * output_delta[j];
        }

        // Actualizar wh: cada hilo actualiza columnas wh[*][j] → strided por j.
        parallel_block(HIDDEN_NEURONS, [&](int j, int /*tid*/) {
            for (int i = 0; i < INPUT_DIM; i++) {
                wh[i][j] -= learning_rate * hidden_delta[j] * input[i];
            }
        });

        // Bias hidden: strided por j.
        parallel_block(HIDDEN_NEURONS, [&](int j, int /*tid*/) {
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

            if ((epoch + 1) % 500 == 0) {
                double mse = total_loss / n_samples;
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
