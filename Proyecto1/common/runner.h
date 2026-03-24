#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "nn_utils.h"

namespace runner {

inline void print_metrics(double mse) {
    std::cout << "MSE:  " << std::fixed << std::setprecision(6) << mse << "\n";
    std::cout << "RMSE: " << std::sqrt(mse) << "\n";
}

template <typename NN>
inline void print_predictions_expected_basefunction(NN& nn,
                                                    const std::vector<std::vector<double>>& X_test,
                                                    int max_print = 5) {
    std::cout << "\nMuestras de predicción:\n";
    for (int i = 0; i < std::min(max_print, (int)X_test.size()); i++) {
        double input[INPUT_DIM];
        for (int j = 0; j < INPUT_DIM; j++) input[j] = X_test[i][j];

        const double pred = nn.predict(input);
        const double expected = basefunction(input, INPUT_DIM);
        std::cout << "Input: [" << input[0] << ", " << input[1] << ", " << input[2]
                  << "] -> Predicción: " << pred
                  << " | Esperado: " << expected << "\n";
    }
}

template <typename NN>
inline void print_predictions_expected_y(NN& nn,
                                        const std::vector<std::vector<double>>& X_test,
                                        const std::vector<double>& Y_test,
                                        int max_print = 5) {
    std::cout << "\nMuestras de predicción:\n";
    for (int i = 0; i < std::min(max_print, (int)X_test.size()); i++) {
        double input[INPUT_DIM];
        for (int j = 0; j < INPUT_DIM; j++) input[j] = X_test[i][j];

        const double pred = nn.predict(input);
        const double expected = Y_test[i];
        std::cout << "Input: [" << input[0] << ", " << input[1] << ", " << input[2]
                  << "] -> Predicción: " << pred
                  << " | Esperado: " << expected << "\n";
    }
}

} // namespace runner
