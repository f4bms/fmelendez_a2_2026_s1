// Red neuronal multihilo con paralelismo real (sin simulación de scheduler).
// A diferencia de CGMT y FGMT, aquí los hilos corren en paralelo verdadero:
// no hay round-robin ni turnos; el SO planifica los hilos libremente.
// No se miden ciclos simulados ni stalls, solo el tiempo de pared.
//
// Distribución de trabajo: scheduling estático strided.
//   El hilo t procesa los índices t, t+T, t+2T, ...
//   Cada neurona/fila es independiente → sin condiciones de carrera.
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
    double output[OUTPUT_DIM];

    int num_threads;

    // Lanza num_threads hilos con scheduling estático strided sobre [0, total_items).
    // fn(idx, tid): idx es el índice de la iteración, tid el id del hilo.
    // Bloquea hasta que todos los hilos terminan.
    template <typename Fn>
    void parallel_strided(int total_items, Fn fn) {
        vector<thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; t++) {
            threads.emplace_back([&, t] {
                for (int idx = t; idx < total_items; idx += num_threads) {
                    fn(idx, t);
                }
            });
        }
        for (auto& th : threads) th.join();
    }

public:
    // Inicialización de Xavier con seed fija para comparaciones reproducibles
    // entre variantes (normal, cgmt, fgmt, threaded con la misma seed).
    explicit NeuralNetworkThreaded(int n_threads, unsigned int fixed_seed = 42u)
        : num_threads(max(1, n_threads)) {
        std::mt19937 rng(fixed_seed);
        std::uniform_real_distribution<double> uni(0.0, 1.0);

        for (int i = 0; i < INPUT_DIM; i++)
            for (int j = 0; j < HIDDEN_NEURONS; j++)
                wh[i][j] = (uni(rng) - 0.5) * sqrt(2.0 / INPUT_DIM);

        for (int j = 0; j < HIDDEN_NEURONS; j++) bh[j] = 0.0;

        for (int i = 0; i < HIDDEN_NEURONS; i++)
            for (int j = 0; j < OUTPUT_DIM; j++)
                wo[i][j] = (uni(rng) - 0.5) * sqrt(2.0 / HIDDEN_NEURONS);

        for (int j = 0; j < OUTPUT_DIM; j++) bo[j] = 0.0;
    }

    // Forward pass en dos etapas.
    // Capa oculta: cada neurona j es independiente → strided en paralelo.
    // Capa de salida: secuencial porque la acumulación sobre hidden_output[]
    //   requeriría sincronización si se paralelizara por hilo de hidden.
    //   Con OUTPUT_DIM=1 el bucle es trivial de todas formas.
    double forward(const double input[]) {
        parallel_strided(HIDDEN_NEURONS, [&](int j, int /*tid*/) {
            double sum = bh[j];
            for (int i = 0; i < INPUT_DIM; i++) {
                sum += input[i] * wh[i][j];
            }
            hidden_input[j] = sum;
            hidden_output[j] = tanh_activation(sum);
        });

        for (int outj = 0; outj < OUTPUT_DIM; outj++) {
            double sum = bo[outj];
            for (int i = 0; i < HIDDEN_NEURONS; i++) {
                sum += hidden_output[i] * wo[i][outj];
            }
            output[outj] = sum;
        }

        return output[0];
    }

    // Backward pass en cinco etapas.
    // Las etapas paralelas son seguras porque cada hilo escribe en índices distintos.
    //
    // 1. hidden_delta: error * tanh'(x) = error * (1 - tanh²(x)), independiente por j.
    // 2. wo -= lr * output_delta * hidden_outputᵀ, independiente por fila i.
    // 3. Bias output: OUTPUT_DIM=1, secuencial.
    // 4. wh -= lr * hidden_delta * inputᵀ, strided por columna j (wh[*][j]).
    // 5. Bias hidden: independiente por j.
    void backward(const double input[], double target, double learning_rate) {
        // Gradiente de MSE: dL/dy = y - target
        double output_delta[OUTPUT_DIM];
        for (int j = 0; j < OUTPUT_DIM; j++) {
            output_delta[j] = output[j] - target;
        }

        // Etapa 1: delta de la capa oculta
        double hidden_delta[HIDDEN_NEURONS];
        parallel_strided(HIDDEN_NEURONS, [&](int j, int /*tid*/) {
            double hidden_error = 0.0;
            for (int k = 0; k < OUTPUT_DIM; k++) {
                hidden_error += output_delta[k] * wo[j][k];
            }
            double t = tanh_activation(hidden_input[j]);
            hidden_delta[j] = hidden_error * (1.0 - t * t);
        });

        // Etapa 2: actualizar wo
        parallel_strided(HIDDEN_NEURONS, [&](int i, int /*tid*/) {
            for (int j = 0; j < OUTPUT_DIM; j++) {
                wo[i][j] -= learning_rate * output_delta[j] * hidden_output[i];
            }
        });

        // Etapa 3: bias de salida (escalar)
        for (int j = 0; j < OUTPUT_DIM; j++) {
            bo[j] -= learning_rate * output_delta[j];
        }

        // Etapa 4: actualizar wh (strided por columna j para evitar races en wh[i][j])
        parallel_strided(HIDDEN_NEURONS, [&](int j, int /*tid*/) {
            for (int i = 0; i < INPUT_DIM; i++) {
                wh[i][j] -= learning_rate * hidden_delta[j] * input[i];
            }
        });

        // Etapa 5: bias oculto
        parallel_strided(HIDDEN_NEURONS, [&](int j, int /*tid*/) {
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
