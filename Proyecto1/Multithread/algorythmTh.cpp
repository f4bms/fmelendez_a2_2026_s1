/* Algoritmo de entrenamiento de una red neuronal con multithreading.
 *
 * Arquitectura: INPUT_DIM → HIDDEN_NEURONS → OUTPUT_DIM  (3 → 70 → 1)
 * Activación:   tanh en la capa oculta, lineal en la salida.
 *
 * Modelo de paralelismo — Scheduling estático por bloques:
 *   - Las HIDDEN_NEURONS neuronas se dividen en bloques contiguos.
 *   - El hilo t procesa el rango [t*block_size, (t+1)*block_size).
 *   - No hay cola de tareas ni sincronización intra-fase; los hilos son
 *     independientes dentro de cada paso (forward / backward).
 *   - Se crean y destruyen hilos en cada llamada a parallel_block().
 *
 * Inicialización: pesos con Xavier usando mt19937 con seed fija (reproducible).
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
    // Pesos y bias: capa entrada→oculta
    double wh[INPUT_DIM][HIDDEN_NEURONS];
    double bh[HIDDEN_NEURONS];

    // Pesos y bias: capa oculta→salida
    double wo[HIDDEN_NEURONS][OUTPUT_DIM];
    double bo[OUTPUT_DIM];

    // Activaciones intermedias (reutilizadas entre forward y backward)
    double hidden_input[HIDDEN_NEURONS];   // suma ponderada antes de tanh
    double hidden_output[HIDDEN_NEURONS];  // salida de tanh por neurona oculta
    double output[OUTPUT_DIM];             // salida final (lineal)

    int num_threads;

    // Lanza `num_threads` hilos con scheduling estático por bloques y espera a que terminen.
    // `fn(idx, tid)` se invoca para cada índice idx en [0, total_items),
    // donde tid es el id del hilo (útil para depuración o métricas).
    // Nota: crear hilos tiene overhead; usar solo cuando total_items es grande.
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
    // Constructor: inicializa pesos con Xavier y bias en cero.
    // Xavier: w ~ Uniforme(-0.5, 0.5) * sqrt(2 / fan_in) para mantener
    // la varianza de las activaciones estable a través de las capas.
    explicit NeuralNetworkThreaded(int n_threads, unsigned int fixed_seed = 42u)
        : num_threads(max(1, n_threads)) {
        // RNG determinista con seed fija (misma semilla que CGMT/FGMT para comparación)
        std::mt19937 rng(fixed_seed);
        std::uniform_real_distribution<double> uni(0.0, 1.0);

        // Pesos wh: fan_in = INPUT_DIM
        for (int i = 0; i < INPUT_DIM; i++) {
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                wh[i][j] = (uni(rng) - 0.5) * sqrt(2.0 / INPUT_DIM);
            }
        }

        for (int j = 0; j < HIDDEN_NEURONS; j++) bh[j] = 0.0;

        // Pesos wo: fan_in = HIDDEN_NEURONS
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            for (int j = 0; j < OUTPUT_DIM; j++) {
                wo[i][j] = (uni(rng) - 0.5) * sqrt(2.0 / HIDDEN_NEURONS);
            }
        }

        for (int j = 0; j < OUTPUT_DIM; j++) bo[j] = 0.0;
    }

    // Propagación hacia adelante. Retorna la predicción output[0].
    //
    // Capa oculta — paralela:
    //   Cada neurona j es independiente (no escribe en celdas compartidas),
    //   por lo que se puede calcular en paralelo sin sincronización.
    //   hidden_input[j]  = bh[j] + Σ_i(input[i] * wh[i][j])
    //   hidden_output[j] = tanh(hidden_input[j])
    //
    // Capa de salida — secuencial:
    //   OUTPUT_DIM=1, no hay beneficio de paralelizar un solo valor.
    //   output[outj] = bo[outj] + Σ_i(hidden_output[i] * wo[i][outj])
    //   (activación lineal: la salida es la suma directa sin tanh)
    double forward(const double input[]) {
        // -- Capa oculta (paralela) --
        parallel_block(HIDDEN_NEURONS, [&](int j, int /*tid*/) {
            double sum = bh[j];
            for (int i = 0; i < INPUT_DIM; i++) {
                sum += input[i] * wh[i][j];
            }
            hidden_input[j]  = sum;
            hidden_output[j] = tanh_activation(sum);
        });

        // -- Capa de salida (secuencial, OUTPUT_DIM=1) --
        for (int outj = 0; outj < OUTPUT_DIM; outj++) {
            double sum = bo[outj];
            for (int i = 0; i < HIDDEN_NEURONS; i++) {
                sum += hidden_output[i] * wo[i][outj];
            }
            output[outj] = sum;  // activación lineal
        }

        return output[0];
    }

    // Retropropagación del error y actualización de pesos (descenso de gradiente).
    //
    // Etapas (en orden):
    //  1. Error de salida:      output_delta[j] = output[j] - target
    //  2. Error capa oculta:    hidden_delta[j] = (Σ_k output_delta[k]*wo[j][k]) * tanh'(hidden_input[j])
    //  3. Actualizar wo:        wo[i][j] -= lr * output_delta[j] * hidden_output[i]
    //  4. Actualizar bo:        bo[j]    -= lr * output_delta[j]           (secuencial)
    //  5. Actualizar wh:        wh[i][j] -= lr * hidden_delta[j] * input[i]
    //  6. Actualizar bh:        bh[j]    -= lr * hidden_delta[j]
    //
    // Las etapas 2, 3, 5 y 6 operan por neurona oculta j de forma independiente → paralelas.
    // Las etapas 1 y 4 son sobre OUTPUT_DIM=1 → secuenciales.
    void backward(const double input[], double target, double learning_rate) {
        // 1. Error de salida (OUTPUT_DIM=1, secuencial)
        double output_delta[OUTPUT_DIM];
        for (int j = 0; j < OUTPUT_DIM; j++) {
            output_delta[j] = output[j] - target;
        }

        // 2. Propagar error a la capa oculta aplicando la derivada de tanh
        //    tanh'(x) = 1 - tanh(x)^2
        double hidden_delta[HIDDEN_NEURONS];
        parallel_block(HIDDEN_NEURONS, [&](int j, int /*tid*/) {
            double hidden_error = 0.0;
            for (int k = 0; k < OUTPUT_DIM; k++) {
                hidden_error += output_delta[k] * wo[j][k];
            }
            double t = tanh_activation(hidden_input[j]);
            hidden_delta[j] = hidden_error * (1.0 - t * t);
        });

        // 3. Actualizar pesos wo (oculta→salida), uno por fila i (paralelo)
        parallel_block(HIDDEN_NEURONS, [&](int i, int /*tid*/) {
            for (int j = 0; j < OUTPUT_DIM; j++) {
                wo[i][j] -= learning_rate * output_delta[j] * hidden_output[i];
            }
        });

        // 4. Actualizar bias de salida (secuencial, OUTPUT_DIM=1)
        for (int j = 0; j < OUTPUT_DIM; j++) {
            bo[j] -= learning_rate * output_delta[j];
        }

        // 5. Actualizar pesos wh (entrada→oculta), columnas wh[*][j] por hilo (paralelo)
        parallel_block(HIDDEN_NEURONS, [&](int j, int /*tid*/) {
            for (int i = 0; i < INPUT_DIM; i++) {
                wh[i][j] -= learning_rate * hidden_delta[j] * input[i];
            }
        });

        // 6. Actualizar bias de la capa oculta (paralelo por j)
        parallel_block(HIDDEN_NEURONS, [&](int j, int /*tid*/) {
            bh[j] -= learning_rate * hidden_delta[j];
        });
    }

    // Entrena la red durante `epochs` épocas.
    // Por cada muestra: forward → calcula error → backward (actualiza pesos).
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
        }
    }

    double predict(const double input[]) { return forward(input); }

    // Evalúa el modelo sobre el conjunto (X, Y) y retorna el MSE (Mean Squared Error).
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
