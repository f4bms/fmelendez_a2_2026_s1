//algoritmo de entrenamiento basico sin hilos

#include <iostream>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <random>

#include "../common/nn_utils.h"
#include "../common/dataset.h"

using namespace std;

// Utilidad para simular stalls de memoria de forma determinista por evento (tag,j,k) y OpType.
// Misma idea que en FGMT/CGMT: probabilidad por tipo de operación, pero el resultado no depende
// del orden de ejecución (stateless).
static inline uint32_t random_det(uint32_t seed, uint32_t tag, int j, int k, OpType op) {
    uint32_t x = seed;
    x ^= tag * 31u;
    x ^= static_cast<uint32_t>(j) * 131u;
    x ^= static_cast<uint32_t>(k) * 197u;
    uint32_t opCode = (op == DOT_PRODUCT) ? 0u : 1u;
    x ^= opCode * 7u;
    x = (x ^ (x >> 13)) * 1274126177u;
    x ^= (x >> 16);
    return x;
}

class NeuralNetwork {
private:
    double wh[INPUT_DIM][HIDDEN_NEURONS]; // pesos de la capa de entrada a la capa oculta
    double bh[HIDDEN_NEURONS]; // bias de la capa oculta
    double wo[HIDDEN_NEURONS][OUTPUT_DIM]; // pesos de la capa oculta a la capa de salida
    double bo[OUTPUT_DIM]; // bias de la capa de salida

    // Valores intermedios para backpropagation
    double hidden_input[HIDDEN_NEURONS];   // Entrada a capa oculta (antes de activación)
    double hidden_output[HIDDEN_NEURONS];  // Salida de capa oculta (después de activación)
    double output_input[OUTPUT_DIM];       // Entrada a capa de salida
    double output[OUTPUT_DIM];             // Salida final

    uint32_t stall_seed;
    std::mt19937 weight_rng;

    // --- Métricas de simulación (sin hilos) ---
    long long cycle_mul_light_count = 0;
    long long cycle_mul_heavy_count = 0;
    long long cycle_mul_count = 0;   // total en ciclos (light+heavy)
    long long total_multiplications = 0; // fuera de ciclos (lr*grad, error^2, etc.)
    long long stall_count = 0;       // cantidad de stalls (NOPs por miss)
    long long global_clock = 0;      // ciclos simulados = cycle_mul_count + stalls (y lo que se modele)

    template <typename T>
    inline T mul_impl(T a, T b, bool is_heavy, OpType op, uint32_t tag, int j, int k) {
        // 1 multiplicación = 1 ciclo de cómputo
        cycle_mul_count++;
        global_clock++;

        if (is_heavy) cycle_mul_heavy_count++;
        else          cycle_mul_light_count++;

        // Stall probabilístico por OpType, determinista por evento.
        // Mismas probabilidades y latencias que FGMT/CGMT.
        // Secuencial: no hay hiding, se paga el stall completo.
        uint32_t miss_pct1000 = (op == DOT_PRODUCT) ? 24u : 6u; // 2.4% → 24/1000, 0.6% → 6/1000
        long long stall_lat  = (op == DOT_PRODUCT) ? 8LL : 3LL; // ciclos
        uint32_t pct = random_det(stall_seed, tag, j, k, op) % 1000u;
        if (pct < miss_pct1000) {
            stall_count++;
            global_clock += stall_lat;
        }

        return a * b;
    }

    // Multiplicaciones en ciclos clasificadas como light/heavy.
    template <typename T>
    inline T mul_light(T a, T b, OpType op, uint32_t tag, int j, int k) {
        return mul_impl(a, b, false, op, tag, j, k);
    }

    template <typename T>
    inline T mul_heavy(T a, T b, OpType op, uint32_t tag, int j, int k) {
        return mul_impl(a, b, true, op, tag, j, k);
    }

    // Multiplicación fuera de ciclos (no consume ciclos, solo se contabiliza como total).
    template <typename T>
    inline T mul_total(T a, T b) {
        total_multiplications++;
        return a * b;
    }
    
public:
    NeuralNetwork(uint32_t seed = 42u) : stall_seed(seed), weight_rng(seed) {
        std::uniform_real_distribution<double> uni(0.0, 1.0);

        //Inicia los pesos al inicio no completamente random se usa una formula para que tenga sentido conforme a la cantidad de entradas y salidas
        for (int i = 0; i < INPUT_DIM; i++) {
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                wh[i][j] = (uni(weight_rng) - 0.5) * sqrt(2.0 / INPUT_DIM);
            }
        }

        // Inicializar bias hidden
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            bh[j] = 0.0;
        }

        // Inicializar pesos de hidden a  output
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            for (int j = 0; j < OUTPUT_DIM; j++) {
                wo[i][j] = (uni(weight_rng) - 0.5) * sqrt(2.0 / HIDDEN_NEURONS);
            }
        }

        // Inicializa bias output
        for (int j = 0; j < OUTPUT_DIM; j++) {
            bo[j] = 0.0;
        }
    }

    // Getters de métricas (útiles para imprimir/CSV desde main)
    long long get_cycle_mul_count() const { return cycle_mul_count; }
    long long get_cycle_mul_light_count() const { return cycle_mul_light_count; }
    long long get_cycle_mul_heavy_count() const { return cycle_mul_heavy_count; }
    long long get_total_multiplications() const { return total_multiplications; }
    long long get_stall_count() const { return stall_count; }
    long long get_global_clock() const { return global_clock; }

    //FORWARDPROPAGATION ---------------------------------------
    double forward(double input[]) {
        // Capa oculta
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            hidden_input[j] = bh[j];
            for (int i = 0; i < INPUT_DIM; i++) {
                hidden_input[j] += mul_light(input[i], wh[i][j], DOT_PRODUCT, 10u, j, i);
            }
            hidden_output[j] = tanh_activation(hidden_input[j]);
        }
        
        // Capa de salida
        for (int j = 0; j < OUTPUT_DIM; j++) {
            output_input[j] = bo[j];
            for (int i = 0; i < HIDDEN_NEURONS; i++) {
                output_input[j] += mul_light(hidden_output[i], wo[i][j], DOT_PRODUCT, 20u, i, j);
            }
            output[j] = output_input[j];
        }
        
        return output[0];
    }
    
    //BACKPROPAGATION ----------------------------------------
    void backward(double input[], double target, double learning_rate) {
        
        //error de salida (derivada de MSE)
        double output_error[OUTPUT_DIM];
        double output_delta[OUTPUT_DIM];
        
        //error medio 
        for (int j = 0; j < OUTPUT_DIM; j++) {
            output_error[j] = output[j] - target;
            output_delta[j] = output_error[j];
        }
        
        //error de capa oculta
        double hidden_error[HIDDEN_NEURONS];
        double hidden_delta[HIDDEN_NEURONS];
        
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            hidden_error[j] = 0.0;
            for (int k = 0; k < OUTPUT_DIM; k++) {
                hidden_error[j] += mul_heavy(output_delta[k], wo[j][k], DOT_PRODUCT, 30u, j, k);
            }

            // ACTIVATION: alinear con FGMT/CGMT (tt = t*t; 1-tt)
            double t = tanh_activation(hidden_input[j]);
            double tt = mul_heavy(t, t, ACTIVATION, 31u, j, 0);
            hidden_delta[j] = mul_heavy(hidden_error[j], (1.0 - tt), ACTIVATION, 32u, j, 0);
        }
        
        //recalculo pesos hidden - output
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            for (int j = 0; j < OUTPUT_DIM; j++) {
                // grad (heavy)
                double grad = mul_heavy(output_delta[j], hidden_output[i], DOT_PRODUCT, 40u, i, j);
                wo[i][j] -= mul_total(learning_rate, grad);
            }
        }
        
        //recalculo bias output
        for (int j = 0; j < OUTPUT_DIM; j++) {
            bo[j] -= mul_total(learning_rate, output_delta[j]);
        }
        
        //recalculo pesos input - hidden
        for (int i = 0; i < INPUT_DIM; i++) {
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                double grad = mul_heavy(hidden_delta[j], input[i], DOT_PRODUCT, 50u, j, i);
                wh[i][j] -= mul_total(learning_rate, grad);
            }
        }
        
        //recalculo bias hidden
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            bh[j] -= mul_total(learning_rate, hidden_delta[j]);
        }
    }
    
    // ENTRENAMIENTO
    void train(vector<vector<double>>& X, vector<double>& Y, int epochs, double learning_rate) {
        int n_samples = X.size();
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double total_loss = 0.0;
            
            for (int i = 0; i < n_samples; i++) {
                double input[INPUT_DIM];
                for (int j = 0; j < INPUT_DIM; j++) {
                    input[j] = X[i][j];
                }
                
                // Forward pass
                double prediction = forward(input);
                
                // Calcular pérdida (MSE)
                double error = prediction - Y[i];
                total_loss += mul_total(error, error);
                
                // Backward pass
                backward(input, Y[i], learning_rate);
            }
            
            // Mostrar progreso cada 100 épocas
            if ((epoch + 1) % 100 == 0) {
                double mse = total_loss / n_samples;
                cout << "Epoch " << (epoch + 1) << "/" << epochs 
                     << " - MSE: " << mse << endl;
            }
        }
    }

    //hace predicción
    double predict(double input[]) {
        return forward(input);
    }
    
    //evalúa que tan efectivo es (MSE)
    double evaluate(vector<vector<double>>& X, vector<double>& Y) {
        double total_loss = 0.0;
        int n_samples = X.size();
        
        for (int i = 0; i < n_samples; i++) {
            double input[INPUT_DIM];
            for (int j = 0; j < INPUT_DIM; j++) {
                input[j] = X[i][j];
            }
            
            double prediction = forward(input);
            double error = prediction - Y[i];
            total_loss += error * error;
        }
        
        return total_loss / n_samples;
    }
    

};