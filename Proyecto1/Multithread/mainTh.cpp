#include "algorythmTh.cpp"
#include "../common/runner.h"

int main(int argc, char* argv[]) {
    int num_threads = 4;
    uint32_t seed = 42u;
    if (argc >= 2) {
        num_threads = max(1, atoi(argv[1]));
    }
    if (argc >= 3) {
        try {
            seed = static_cast<uint32_t>(std::stoul(argv[2]));
        } catch (...) {
            return 1;
        }
    }

    vector<vector<double>> X_all;
    vector<double> Y_all;
    if (!loadDataset("../dataset.txt", X_all, Y_all) || X_all.empty()) {
        cerr << "Error: el dataset está vacío o no encontrado." << endl;
        return 1;
    }

    auto split = split_train_test(X_all, Y_all, 0.8);

    NeuralNetworkThreaded nn(num_threads, seed);
    nn.train(split.X_train, split.Y_train, EPOCHS, LEARNING_RATE);

    double mse = nn.evaluate(split.X_test, split.Y_test);
    cout << mse << endl;

    return 0;
}
