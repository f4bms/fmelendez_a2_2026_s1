#pragma once

#include <cmath>

// Funciones utilitarias compartidas entre implementaciones.

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
