#pragma once

#include <cmath>

// --- Configuración base compartida para todas las variantes ---
constexpr int INPUT_DIM = 3;
constexpr int HIDDEN_NEURONS = 70;
constexpr int OUTPUT_DIM = 1;
constexpr double LEARNING_RATE = 0.08;
constexpr int EPOCHS = 500;

// --- Tipo de operación para clasificar latencia en el scheduler ---
// DOT_PRODUCT : multiplicaciones de pesos/gradientes (w*x, w*delta, ...)
// ACTIVATION  : multiplicaciones dentro del cómputo de la activación (t*t, error*deriv, ...)
enum OpType {
    DOT_PRODUCT,
    ACTIVATION
};

// --- Funciones matemáticas útiles para la red neuronal ---
inline double basefunction(const double x[], int d) {
    double sum = 0.0;
    for (int i = 0; i < d; i++) {
        sum += std::sin(x[i]) + 0.3 * x[i] * x[i];
    }
    return sum;
}

inline double tanh_activation(double x) {
    return std::tanh(x);
}

inline double tanh_derivative(double x) {
    double t = std::tanh(x);
    return 1.0 - t * t;
}