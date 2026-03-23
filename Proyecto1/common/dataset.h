#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Carga dataset desde un txt con formato: x1 x2 x3 y
inline bool loadDataset(const std::string& filename,
                        std::vector<std::vector<double>>& X,
                        std::vector<double>& Y) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo de datos: " << filename << std::endl;
        return false;
    }

    double x1, x2, x3, y;
    while (file >> x1 >> x2 >> x3 >> y) {
        X.push_back({x1, x2, x3});
        Y.push_back(y);
    }

    std::cout << "Dataset cargado: " << X.size() << " muestras" << std::endl;
    return true;
}

struct TrainTestSplit {
    std::vector<std::vector<double>> X_train;
    std::vector<double> Y_train;
    std::vector<std::vector<double>> X_test;
    std::vector<double> Y_test;
};

inline TrainTestSplit split_train_test(const std::vector<std::vector<double>>& X_all,
                                       const std::vector<double>& Y_all,
                                       double train_ratio = 0.8) {
    TrainTestSplit s;
    const int n_train = static_cast<int>(X_all.size() * train_ratio);
    s.X_train.assign(X_all.begin(), X_all.begin() + n_train);
    s.Y_train.assign(Y_all.begin(), Y_all.begin() + n_train);
    s.X_test.assign(X_all.begin() + n_train, X_all.end());
    s.Y_test.assign(Y_all.begin() + n_train, Y_all.end());
    return s;
}
