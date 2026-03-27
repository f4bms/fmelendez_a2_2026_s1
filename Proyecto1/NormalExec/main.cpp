#include "algorythm.cpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include <filesystem>

#include "../common/runner.h"

static void append_metrics_csv(const std::string& tipo,
                               int cant_threads,
                               int input_dim,
                               int hidden_neurons,
                               int output_dim,
                               int epochs,
                               int dataset_elements,
                               long long total_mult_en_ciclos,
                               long long nops_stalls,
                               long long ciclos_simulados,
                               long long context_switches,
                               double tiempo_ejec_s) {
    namespace fs = std::filesystem;
    const fs::path outPath = fs::current_path() / "metrics.csv";

    const bool needHeader = (!fs::exists(outPath)) || (fs::file_size(outPath) == 0);
    std::ofstream out(outPath, std::ios::app);
    if (!out) return;

    if (needHeader) {
        out << "tipo,cant_threads,input_dim,hidden_neurons,output_dim,epochs,dataset_elements,total_mult_en_ciclos,nops_stalls,ciclos_simulados,context_switches,tiempo_ejecucion_s\n";
    }

    out << tipo << ','
        << cant_threads << ','
        << input_dim << ','
        << hidden_neurons << ','
        << output_dim << ','
        << epochs << ','
        << dataset_elements << ','
        << total_mult_en_ciclos << ','
        << nops_stalls << ','
        << ciclos_simulados << ','
        << context_switches << ','
        << std::fixed << std::setprecision(6) << tiempo_ejec_s
        << "\n";
}

int main(int argc, char* argv[]) {
    uint32_t seed = 42u;
    if (argc >= 2) {
        try {
            seed = static_cast<uint32_t>(std::stoul(argv[1]));
        } catch (...) {
            return 1;
        }
    }

    // Cargar todos los datos
    vector<vector<double>> X_all;
    vector<double> Y_all;
    if (!loadDataset("../dataset.txt", X_all, Y_all) || X_all.empty()) {
        return 1;
    }

    auto split = split_train_test(X_all, Y_all, 0.8);
    auto& X_train = split.X_train;
    auto& Y_train = split.Y_train;
    auto& X_test  = split.X_test;
    auto& Y_test  = split.Y_test;

    NeuralNetwork nn(seed);

    auto start = std::chrono::high_resolution_clock::now();

    nn.train(X_train, Y_train, EPOCHS, LEARNING_RATE);

    // (silencioso) evaluamos para mantener el mismo flujo
    (void)nn.evaluate(X_test, Y_test);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    // CSV metrics (normal no tiene context switches)
    append_metrics_csv(
        "normal",
        /*threads*/ 0,
        INPUT_DIM,
        HIDDEN_NEURONS,
        OUTPUT_DIM,
        EPOCHS,
        (int)X_all.size(),
        /*total_mult_en_ciclos*/ nn.get_cycle_mul_count(),
        /*nops_stalls*/ nn.get_stall_count(),
        /*ciclos_simulados*/ nn.get_global_clock(),
        /*context_switches*/ 0,
        /*tiempo_ejec_s*/ duration.count());

    return 0;
}
