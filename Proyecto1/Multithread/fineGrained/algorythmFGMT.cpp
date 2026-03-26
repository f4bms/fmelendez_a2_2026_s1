/* Algoritmo de entrenamiento de una red neuronal utilizando fine grained multithreading
-simulacion de ejecución de ciclos por multiplicaciones(3 tipos: livianas, pesadas y fuera de ciclos)
-round robin scheduler: alterna entre hilos en cada multiplicación
*/
#include <iostream>
#include <cstdlib>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <chrono>
#include <iomanip>
#include <random>
#include "../../common/nn_utils.h"
#include "../../common/dataset.h"

using namespace std;
using namespace std::chrono;

// scheduler de round robin
// distribuido por neuronas y ciclos establecidos como multiplicaciones
class RoundRobinScheduler {
private:
    int num_threads;
    long long context_switch_penalty_cycles;
    long long light_count;
    long long heavy_count;
    long long light_dot_count;
    long long light_act_count;
    long long heavy_dot_count;
    long long heavy_act_count;
    long long fetch_count;
    long long stall_nop_count;
    long long context_switch_count;
    long long total_multiplications;
    long long global_clock;
    int current_turn;

    
    static const long long LATENCY_LIGHT_DOT    = 1;   // forward: w*x  (pipelined, liviano)
    static const long long LATENCY_LIGHT_ACT    = 1;   // (reservado para activaciones ligeras futuras)
    static const long long LATENCY_HEAVY_DOT    = 3;   // backward: w*delta  (gradiente, pesado)
    static const long long LATENCY_HEAVY_ACT    = 2;   // backward: t*t, error*deriv  (derivada de tanh, más costoso)

    mutex m;
    condition_variable cv;
    bool phase_open;

    // Pseudo-random determinista (stateless) para poder reproducir el mismo patrón
    // de stalls entre ejecuciones y (eventualmente) entre esquemas.
    static inline uint32_t random(uint32_t seed, uint32_t tag, int j, int k, OpType op) {
        uint32_t x = seed;
    // Constantes pequeñas para que sea fácil de seguir/explicar (aún determinista).
    x ^= tag * 31u;
    x ^= static_cast<uint32_t>(j) * 131u;
    x ^= static_cast<uint32_t>(k) * 197u;
        uint32_t opCode = (op == DOT_PRODUCT) ? 0u : 1u;
    x ^= opCode * 7u;

    x = (x ^ (x >> 13)) * 1274126177u;
    x ^= (x >> 16);
        return x;
    }

public:
    RoundRobinScheduler(int num_threads_arg, long long context_switch_penalty_cycles_arg = 1)
    : num_threads(num_threads_arg),
    context_switch_penalty_cycles(context_switch_penalty_cycles_arg),
    light_count(0), heavy_count(0),
    light_dot_count(0), light_act_count(0),
    heavy_dot_count(0), heavy_act_count(0),
    fetch_count(0), stall_nop_count(0),
    context_switch_count(0),
    total_multiplications(0), global_clock(0), current_turn(0),
    phase_open(false) {}

    void begin_phase() {
        unique_lock<mutex> lock(m);
        phase_open = true;
        current_turn = 0;
        cv.notify_all();
    }

    void end_phase() {
        unique_lock<mutex> lock(m);
        phase_open = false;
        cv.notify_all();
    }

    bool is_phase_open() {
        lock_guard<mutex> lock(m);
        return phase_open;
    }

    // espera el turno del hilo y retorna true si la fase sigue abierta
    bool wait_turn(int thread_id) {
        unique_lock<mutex> lock(m);
        cv.wait(lock, [&] { return !phase_open || current_turn == thread_id; });
        return phase_open;
    }

    // En scheduling estático puede pasar que un hilo termine antes que otro (porque tiene
    // menos neuronas asignadas). Para evitar deadlock, este helper hace que el hilo siga
    // cediendo turnos hasta que la fase se cierre.
    void yield_turns_until_closed(int thread_id) {
        while (true) {
            if (!wait_turn(thread_id)) return;
            advance_turn();
        }
    }

    // avanza el turno
    void advance_turn() {
        unique_lock<mutex> lock(m);
        const int prev_turn = current_turn;
        current_turn = (current_turn + 1) % num_threads;
        cv.notify_all();
    }

private:
    void apply_context_switch_latency_locked() {
        if (num_threads <= 1) return;
        context_switch_count++;
        global_clock += context_switch_penalty_cycles;
    }

    // Selecciona la latencia correcta según si la operación es light/heavy y su OpType.
    long long pick_latency(bool is_heavy, OpType op) const {
        if (!is_heavy) {
            return (op == ACTIVATION) ? LATENCY_LIGHT_ACT : LATENCY_LIGHT_DOT;
        } else {
            return (op == ACTIVATION) ? LATENCY_HEAVY_ACT : LATENCY_HEAVY_DOT;
        }
    }

public:
    // En esta versión (estática) no hay "fetch" dinámico.
    // Dejamos el contador en 0 para comparabilidad de métricas.

    void count_light() { lock_guard<mutex> l(m); light_count++; }
    void count_heavy() { lock_guard<mutex> l(m); heavy_count++; }

    // op indica si es una multiplicación de producto punto (DOT_PRODUCT) o
    // parte del cómputo de activación (ACTIVATION), determinando la latencia de context switch.
    template <typename T>
    T mul_light(T a, T b, OpType op) {
        lock_guard<mutex> lock(m);
        light_count++;
    if (op == DOT_PRODUCT) light_dot_count++;
    else                  light_act_count++;
        global_clock++;
        T r = a * b;
        const int prev_turn = current_turn;
        current_turn = (current_turn + 1) % num_threads;
        if (num_threads > 1 && current_turn != prev_turn) {
            apply_context_switch_latency_locked();
        }
        cv.notify_all();
        return r;
    }

    template <typename T>
    T mul_heavy(T a, T b, OpType op) {
        lock_guard<mutex> lock(m);
        heavy_count++;
    if (op == DOT_PRODUCT) heavy_dot_count++;
    else                  heavy_act_count++;
        global_clock++;
        T r = a * b;
        const int prev_turn = current_turn;
        current_turn = (current_turn + 1) % num_threads;
        if (num_threads > 1 && current_turn != prev_turn) {
            apply_context_switch_latency_locked();
        }
        cv.notify_all();
        return r;
    }

    // Simula stalls (misses) de memoria dependiendo del tipo de operación.
    // Retorna false si la fase se cerró mientras el hilo esperaba su turno.
    // tag/j/k identifican de forma determinista el "evento" (multiplicación lógica)
    // para que el patrón de stalls sea reproducible y no dependa del scheduling.
    bool check_stall(int thread_id, OpType op, uint32_t tag, int j, int k) {
        // DOT_PRODUCT: alta presión de memoria → mayor probabilidad y stall más largo.
        // ACTIVATION: compute-bound → menor probabilidad y stall corto.
        double miss_prob    = (op == DOT_PRODUCT) ? 2.4 : 0.6;  // %
        long long stall_lat = (op == DOT_PRODUCT) ? 8LL  : 3LL; // ciclos del stall

        static constexpr uint32_t SEED = 42u;
        (void)thread_id;
        uint32_t pct = random(SEED, tag, j, k, op) % 100u;

        if (pct < static_cast<uint32_t>(miss_prob)) {
            lock_guard<mutex> l(m);
            if (!phase_open) return false;
            stall_nop_count++;
            // FGMT: el pipeline siempre tiene num_threads slots. Un hilo terminado
            // ocupa su slot con NOPs, así que la capacidad de absorción es siempre num_threads-1.
            long long absorbed = std::min(stall_lat, (long long)(num_threads - 1));
            global_clock += std::max(0LL, stall_lat - absorbed);
        }

        return true;
    }

    // Multiplicación fuera de ciclos (lr * grad, error^2, etc.).
    template <typename T>
    T mul_total(T a, T b) {
        lock_guard<mutex> lock(m);
        total_multiplications++;
        return a * b;
    }

    long long get_light_count()           const { return light_count; }
    long long get_heavy_count()           const { return heavy_count; }
    long long get_light_dot_count()       const { return light_dot_count; }
    long long get_light_act_count()       const { return light_act_count; }
    long long get_heavy_dot_count()       const { return heavy_dot_count; }
    long long get_heavy_act_count()       const { return heavy_act_count; }
    long long get_stall_nop_count()       const { return stall_nop_count; }
    long long get_context_switch_count()  const { return context_switch_count; }
    long long get_total_multiplications() const { return total_multiplications; }
    long long get_global_clock()          const { return global_clock; }
    long long get_fetch_count()           const { return fetch_count; }
    int       get_num_threads()           const { return num_threads; }

    void print_stats() {
        long long cycle_muls = light_count + heavy_count;
        cout << "\n --- metricas ---" << endl;
    cout << "mult light:  " << light_count          << endl;
    cout << "  - light DOT: " << light_dot_count    << endl;
    cout << "  - light ACT: " << light_act_count    << endl;
    cout << "mult heavy:  " << heavy_count          << endl;
    cout << "  - heavy DOT: " << heavy_dot_count    << endl;
    cout << "  - heavy ACT: " << heavy_act_count    << endl;
    cout << "fetches (take_next): " << fetch_count  << endl;
        cout << "mult fuera de ciclos: " << total_multiplications << endl;
        cout << "total mult en ciclos: " << cycle_muls           << endl;
        cout << "context switches:               " << context_switch_count  << endl;
        cout << "NOPs por stall:                   " << stall_nop_count      << endl;
        cout << "ciclos simulados:  " << global_clock         << endl;
        cout << "----------------------\n" << endl;
    }
};


class NeuralNetworkFineGrained {
private:
    double wh[INPUT_DIM][HIDDEN_NEURONS];
    double bh[HIDDEN_NEURONS];
    double wo[HIDDEN_NEURONS][OUTPUT_DIM];
    double bo[OUTPUT_DIM];
    
    double hidden_input[HIDDEN_NEURONS];
    double hidden_output[HIDDEN_NEURONS];
    double output_input[OUTPUT_DIM];
    double output[OUTPUT_DIM];
    
    RoundRobinScheduler* scheduler;

    //se usa un generador de numeros aleatorios
    std::mt19937 rng;

    double uniform_symmetric(double scale) {
        std::uniform_real_distribution<double> dist(-scale, scale);
        return dist(rng);
    }
    
public:
    NeuralNetworkFineGrained(RoundRobinScheduler* sched) : scheduler(sched) {
    // Semilla fija para reproducibilidad entre ejecuciones.
    // Si querés variabilidad controlada, podés convertirlo en parámetro desde main.
    static constexpr uint32_t NN_SEED = 12345u;
    rng.seed(NN_SEED);
        
        // base inicial de pesos de input a hidden
        for (int i = 0; i < INPUT_DIM; i++) {
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                wh[i][j] = uniform_symmetric(0.5) * sqrt(2.0 / INPUT_DIM);
            }
        }

        // base inicial de bias hidden
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            bh[j] = 0.0;
        }

        // base inicial de pesos de hidden a output
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            for (int j = 0; j < OUTPUT_DIM; j++) {
                wo[i][j] = uniform_symmetric(0.5) * sqrt(2.0 / HIDDEN_NEURONS);
            }
        }

        // base inicial de bias output
        for (int j = 0; j < OUTPUT_DIM; j++) {
            bo[j] = 0.0;
        }
    }
    
    // Forward propagation ------------------------------------------------------
    double forward_finegrained(const double input[]) {
        vector<thread> threads;
        int num_threads = scheduler->get_num_threads();

        // Scheduling estático: cada hilo procesa neuronas j = thread_id, thread_id+T, ...
        // El scheduler solo se usa para orden de ejecución (turno) y contabilizar ciclos/latencias.
    scheduler->begin_phase();
    std::atomic<int> hidden_finished{0};
        auto compute_hidden = [&](int thread_id) {
            for (int j = thread_id; j < HIDDEN_NEURONS; j += num_threads) {
                hidden_input[j] = bh[j];
                for (int k = 0; k < INPUT_DIM; k++) {
                    if (!scheduler->wait_turn(thread_id)) return;
                    hidden_input[j] += scheduler->mul_light(input[k], wh[k][j], DOT_PRODUCT);
                    scheduler->advance_turn();
                    if (!scheduler->check_stall(thread_id, DOT_PRODUCT, /*tag*/ 10u, /*j*/ j, /*k*/ k)) return;
                }
                hidden_output[j] = tanh_activation(hidden_input[j]);
            }
            if (hidden_finished.fetch_add(1) + 1 == num_threads) {
                scheduler->end_phase();
            }
            scheduler->yield_turns_until_closed(thread_id);
        };

        for (int t = 0; t < num_threads; t++) {
            threads.emplace_back(compute_hidden, t);
        }
        for (auto& t : threads) {
            t.join();
        }

        //comienzo de la capa de salida
    scheduler->begin_phase();
    std::atomic<int> output_finished{0};
        for (int outj = 0; outj < OUTPUT_DIM; outj++) {
            output_input[outj] = bo[outj];
        }

        auto compute_output = [&](int thread_id) {
            for (int i = thread_id; i < HIDDEN_NEURONS; i += num_threads) {
                for (int outj = 0; outj < OUTPUT_DIM; outj++) {
                    if (!scheduler->wait_turn(thread_id)) return;
                    output_input[outj] += scheduler->mul_light(hidden_output[i], wo[i][outj], DOT_PRODUCT);
                    scheduler->advance_turn();
                    if (!scheduler->check_stall(thread_id, DOT_PRODUCT, /*tag*/ 20u, /*j*/ i, /*k*/ outj)) return;
                }
            }
            if (output_finished.fetch_add(1) + 1 == num_threads) {
                scheduler->end_phase();
            }
            scheduler->yield_turns_until_closed(thread_id);
        };

        threads.clear();
        for (int t = 0; t < num_threads; t++) {
            threads.emplace_back(compute_output, t);
        }
        for (auto& t : threads) {
            t.join();
        }

        for (int outj = 0; outj < OUTPUT_DIM; outj++) {
            output[outj] = output_input[outj];
        }
        
        return output[0];
    }
                                                
    //BACKPROPAGATION ------------------------------------------------------------
    void backward_finegrained(const double input[], double target, double learning_rate) {
        
        // Gradiente de MSE respecto a la salida
        double output_delta[OUTPUT_DIM];
        for (int j = 0; j < OUTPUT_DIM; j++) {
            output_delta[j] = output[j] - target;
        }
        
        double hidden_error[HIDDEN_NEURONS];
        double hidden_delta[HIDDEN_NEURONS];

        int num_threads = scheduler->get_num_threads();
        scheduler->begin_phase();
    std::atomic<int> bhe_finished{0};
        vector<thread> threads;
        auto compute_hidden_error = [&](int thread_id) {
            for (int j = thread_id; j < HIDDEN_NEURONS; j += num_threads) {
                hidden_error[j] = 0.0;
                for (int k = 0; k < OUTPUT_DIM; k++) {
                    if (!scheduler->wait_turn(thread_id)) return;
                    hidden_error[j] += scheduler->mul_heavy(output_delta[k], wo[j][k], DOT_PRODUCT);
                    scheduler->advance_turn();
                    if (!scheduler->check_stall(thread_id, DOT_PRODUCT, /*tag*/ 30u, /*j*/ j, /*k*/ k)) return;
                }

                // Derivada tanh: (1 - t^2)
                if (!scheduler->wait_turn(thread_id)) return;
                double t = tanh(hidden_input[j]);
                double tt = scheduler->mul_heavy(t, t, ACTIVATION);
                scheduler->advance_turn();
                if (!scheduler->check_stall(thread_id, ACTIVATION, /*tag*/ 31u, /*j*/ j, /*k*/ 0)) return;

                if (!scheduler->wait_turn(thread_id)) return;
                hidden_delta[j] = scheduler->mul_heavy(hidden_error[j], (1.0 - tt), ACTIVATION);
                scheduler->advance_turn();
                if (!scheduler->check_stall(thread_id, ACTIVATION, /*tag*/ 32u, /*j*/ j, /*k*/ 0)) return;
            }
            if (bhe_finished.fetch_add(1) + 1 == num_threads) {
                scheduler->end_phase();
            }
            scheduler->yield_turns_until_closed(thread_id);
        };

    for (int t = 0; t < num_threads; t++) threads.emplace_back(compute_hidden_error, t);
    for (auto& t : threads) t.join();

    // Actualización de pesos hidden -> output
    scheduler->begin_phase();
    std::atomic<int> uwo_finished{0};
        threads.clear();
        auto update_wo = [&](int thread_id) {
            for (int i = thread_id; i < HIDDEN_NEURONS; i += num_threads) {
                for (int outj = 0; outj < OUTPUT_DIM; outj++) {
                    if (!scheduler->wait_turn(thread_id)) return;
                    double grad = scheduler->mul_heavy(output_delta[outj], hidden_output[i], DOT_PRODUCT);
                    wo[i][outj] -= scheduler->mul_total(learning_rate, grad);
                    scheduler->advance_turn();
                    if (!scheduler->check_stall(thread_id, DOT_PRODUCT, /*tag*/ 40u, /*j*/ i, /*k*/ outj)) return;
                }
            }
            if (uwo_finished.fetch_add(1) + 1 == num_threads) {
                scheduler->end_phase();
            }
            scheduler->yield_turns_until_closed(thread_id);
        };
    for (int t = 0; t < num_threads; t++) threads.emplace_back(update_wo, t);
    for (auto& t : threads) t.join();
        
    // Bias de salida (fuera de ciclos)
        for (int j = 0; j < OUTPUT_DIM; j++) {
            bo[j] -= scheduler->mul_total(learning_rate, output_delta[j]);
        }
        
    // Actualización de pesos input -> hidden
    scheduler->begin_phase();
    std::atomic<int> uwh_finished{0};
        threads.clear();
        auto update_wh = [&](int thread_id) {
            for (int j = thread_id; j < HIDDEN_NEURONS; j += num_threads) {
                for (int k = 0; k < INPUT_DIM; k++) {
                    if (!scheduler->wait_turn(thread_id)) return;
                    double grad = scheduler->mul_heavy(hidden_delta[j], input[k], DOT_PRODUCT);
                    wh[k][j] -= scheduler->mul_total(learning_rate, grad);
                    scheduler->advance_turn();
                    if (!scheduler->check_stall(thread_id, DOT_PRODUCT, /*tag*/ 50u, /*j*/ j, /*k*/ k)) return;
                }
            }
            if (uwh_finished.fetch_add(1) + 1 == num_threads) {
                scheduler->end_phase();
            }
            scheduler->yield_turns_until_closed(thread_id);
        };
    for (int t = 0; t < num_threads; t++) threads.emplace_back(update_wh, t);
    for (auto& t : threads) t.join();
        
    // Bias de capa oculta (fuera de ciclos)
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            bh[j] -= scheduler->mul_total(learning_rate, hidden_delta[j]);
        }
    }
    
    // entrenamiento y predicción
    void train(const vector<vector<double>>& X, const vector<double>& Y, 
               int epochs, double learning_rate) {
        int n_samples = X.size();
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double total_loss = 0.0;
            
            for (int i = 0; i < n_samples; i++) {
                double input[INPUT_DIM];
                for (int j = 0; j < INPUT_DIM; j++) {
                    input[j] = X[i][j];
                }
                
                double prediction = forward_finegrained(input);
                double error = prediction - Y[i];
                total_loss += scheduler->mul_total(error, error);
                
                backward_finegrained(input, Y[i], learning_rate);
            }
            
            if ((epoch + 1) % 500 == 0) {
                double mse = total_loss / n_samples;
                cout << "Epoch " << (epoch + 1) << "/" << epochs 
                     << " - MSE: " << mse << endl;
            }
        }
    }
    
    double predict(const double input[]) {
        return forward_finegrained(input);
    }
    
    double evaluate(const vector<vector<double>>& X, const vector<double>& Y) {
        double total_loss = 0.0;
        int n_samples = X.size();
        
        for (int i = 0; i < n_samples; i++) {
            double input[INPUT_DIM];
            for (int j = 0; j < INPUT_DIM; j++) {
                input[j] = X[i][j];
            }
            
            double prediction = forward_finegrained(input);
            double error = prediction - Y[i];
            total_loss += scheduler->mul_total(error, error);
        }
        
        return total_loss / n_samples;
    }
    
    RoundRobinScheduler* get_scheduler() { return scheduler; }
};

// loadDataset(...) movido a common/dataset.h