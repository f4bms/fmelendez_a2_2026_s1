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
    long long light_count;
    long long heavy_count;
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
    int phase_done_count;
    bool phase_open;

    atomic<int> next_neuron;

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
    RoundRobinScheduler(int num_threads_arg, long long /*quantum_size*/ = 1)
    : num_threads(num_threads_arg),
    light_count(0), heavy_count(0), fetch_count(0), stall_nop_count(0),
    context_switch_count(0),
    total_multiplications(0), global_clock(0), current_turn(0),
      phase_done_count(0), phase_open(false),
    next_neuron(0) {}

    void begin_phase() {
        next_neuron.store(0, memory_order_relaxed);
        unique_lock<mutex> lock(m);
        phase_open = true;
        phase_done_count = 0;
        current_turn = 0;
        cv.notify_all();
    }

    // espera el turno del hilo y retorna true si la fase sigue abierta
    bool wait_turn(int thread_id) {
        unique_lock<mutex> lock(m);
        cv.wait(lock, [&] { return !phase_open || current_turn == thread_id; });
        return phase_open;
    }

    // avanza el turno
    void advance_turn() {
        unique_lock<mutex> lock(m);
        global_clock++;
        const int prev_turn = current_turn;
        current_turn = (current_turn + 1) % num_threads;
        cv.notify_all();
    }

private:
    void apply_context_switch_latency_locked(long long latency_cycles) {
        if (num_threads <= 1) return;
        context_switch_count++;
        global_clock += latency_cycles;
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

    // Registra que este hilo terminó su trabajo, solo cierra la fase cuando ya todos terminaron
    void mark_done() {
        unique_lock<mutex> lock(m);
        if (!phase_open) return;
        phase_done_count++;
        if (phase_done_count >= num_threads) phase_open = false;
        cv.notify_all();
    }

    // Toma el siguiente índice de neurona disponible y retorna -1 si no hay más
    int take_next(int total) {
        // Simula el costo de hacer fetch de la cola (1 ciclo) y lo hace determinista.
        {
            lock_guard<mutex> lock(m);
            fetch_count++;
            global_clock += 1;
        }
        int idx = next_neuron.fetch_add(1, memory_order_relaxed);
        return (idx < total) ? idx : -1;
    }

    void count_light() { lock_guard<mutex> l(m); light_count++; }
    void count_heavy() { lock_guard<mutex> l(m); heavy_count++; }

    // op indica si es una multiplicación de producto punto (DOT_PRODUCT) o
    // parte del cómputo de activación (ACTIVATION), determinando la latencia de context switch.
    template <typename T>
    T mul_light(T a, T b, OpType op) {
        lock_guard<mutex> lock(m);
        light_count++;
        global_clock++;
        T r = a * b;
        const int prev_turn = current_turn;
        current_turn = (current_turn + 1) % num_threads;
        if (num_threads > 1 && current_turn != prev_turn) {
            apply_context_switch_latency_locked(pick_latency(false, op));
        }
        cv.notify_all();
        return r;
    }

    template <typename T>
    T mul_heavy(T a, T b, OpType op) {
        lock_guard<mutex> lock(m);
        heavy_count++;
        global_clock++;
        T r = a * b;
        const int prev_turn = current_turn;
        current_turn = (current_turn + 1) % num_threads;
        if (num_threads > 1 && current_turn != prev_turn) {
            apply_context_switch_latency_locked(pick_latency(true, op));
        }
        cv.notify_all();
        return r;
    }

    // Simula stalls (misses) de memoria dependiendo del tipo de operación.
    // Retorna false si la fase se cerró mientras el hilo esperaba su turno.
    // tag/j/k identifican de forma determinista el "evento" (multiplicación lógica)
    // para que el patrón de stalls sea reproducible y no dependa del scheduling.
    bool check_stall(int thread_id, OpType op, uint32_t tag, int j, int k) {
        double base_miss_rate = 2.0; // % base
        double pressure = (op == DOT_PRODUCT) ? 1.2 : 0.3; // DOT_PRODUCT tiende a presionar más memoria
        double miss_prob = base_miss_rate * pressure;      // % final

    // Por ahora, mantenemos el patrón fijo por hilo (thread_id) para no cambiar
    // las firmas de llamadas en todo el código. Si luego querés consistencia
    // entre esquemas, se puede extender con (j,k) lógicos de la operación.
    static constexpr uint32_t SEED = 42u;
    (void)thread_id; // thread_id se mantiene por compatibilidad/telemetría
    uint32_t pct = random(SEED, tag, j, k, op) % 100u;

        // Importante: check_stall() se llama desde secciones donde el hilo YA obtuvo
        // su turno vía wait_turn(thread_id). Volver a hacer wait_turn aquí puede
        // bloquearse (deadlock) porque current_turn podría ya haber avanzado.
        // En stall solo consumimos ciclos (NOP) y cedemos el turno.
        if (pct < static_cast<uint32_t>(miss_prob)) {
            {
                lock_guard<mutex> l(m);
                if (!phase_open) return false;
                stall_nop_count++;
                global_clock += 1; // penalidad fija por miss (ciclos)
            }
            advance_turn();
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
        cout << "mult heavy:  " << heavy_count          << endl;
    cout << "fetches (take_next): " << fetch_count     << endl;
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

    //el scheduler se realiza por neuronas, dentro de cada una se realizan multiplicaciones
    //en el forward las multiplicaciones son contadas como light
    //3 cases por hilo: FETCH, WORK, y IDLE
    scheduler->begin_phase();
        auto compute_hidden = [&](int thread_id) {
            int cur_neuron = -1, cur_dim = 0;
            bool idle = false;
            while (true) {
                if (!scheduler->wait_turn(thread_id)) break;

                if (idle) {
                    scheduler->advance_turn();

                } else if (cur_neuron < 0) {
                    int j = scheduler->take_next(HIDDEN_NEURONS);
                    if (j < 0) {
                        scheduler->mark_done();
                        idle = true;
                    } else {
                        cur_neuron = j;
                        cur_dim = 0;
                        hidden_input[j] = bh[j];
                    }
                    scheduler->advance_turn();
                
                } else {
                    // w * x: producto punto puro → DOT_PRODUCT
                    hidden_input[cur_neuron] += scheduler->mul_light(input[cur_dim], wh[cur_dim][cur_neuron], DOT_PRODUCT);
                    int stall_j = cur_neuron;
                    int stall_k = cur_dim;
                    cur_dim++;
                    if (cur_dim >= INPUT_DIM) {
                        hidden_output[cur_neuron] = tanh_activation(hidden_input[cur_neuron]);
                        cur_neuron = -1;
                    }
                    scheduler->advance_turn();
                    if (!scheduler->check_stall(thread_id, DOT_PRODUCT, /*tag*/ 10u, stall_j, stall_k)) break;
                }
            }
        };

        for (int t = 0; t < num_threads; t++) {
            threads.emplace_back(compute_hidden, t);
        }
        for (auto& t : threads) {
            t.join();
        }

        //comienzo de la capa de salida
        scheduler->begin_phase();
        for (int outj = 0; outj < OUTPUT_DIM; outj++) {
            output_input[outj] = bo[outj];
        }

        auto compute_output = [&](int thread_id) {
            int cur_neuron = -1, cur_out = 0;
            bool idle = false;
            while (true) {
                if (!scheduler->wait_turn(thread_id)) break;

                if (idle) {
                    scheduler->advance_turn();

                } else if (cur_neuron < 0) {
                    int i = scheduler->take_next(HIDDEN_NEURONS);
                    if (i < 0) {
                        scheduler->mark_done(); idle = true;
                    } else {
                        cur_neuron = i; cur_out = 0;
                    }
                    scheduler->advance_turn();

                } else {
                    // hidden_out * wo: producto punto puro → DOT_PRODUCT
                    output_input[cur_out] += scheduler->mul_light(hidden_output[cur_neuron], wo[cur_neuron][cur_out], DOT_PRODUCT);
                    int stall_j = cur_neuron;
                    int stall_k = cur_out;
                    cur_out++;
                    if (cur_out >= OUTPUT_DIM) cur_neuron = -1;
                    scheduler->advance_turn();
                    if (!scheduler->check_stall(thread_id, DOT_PRODUCT, /*tag*/ 20u, stall_j, stall_k)) break;
                }
            }
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
        vector<thread> threads;
        auto compute_hidden_error = [&](int thread_id) {
            int cur_neuron = -1, cur_phase = 0, cur_k = 0;
            bool idle = false;
            while (true) {
                if (!scheduler->wait_turn(thread_id)) break;

                if (idle) {
                    scheduler->advance_turn();

                } else if (cur_neuron < 0) {
                    // FETCH: ciclo NOP de despacho
                    int j = scheduler->take_next(HIDDEN_NEURONS);
                    if (j < 0) {
                        scheduler->mark_done(); idle = true;
                    } else {
                        cur_neuron = j; cur_phase = 0; cur_k = 0;
                        hidden_error[j] = 0.0;
                    }
                    scheduler->advance_turn();

                } else if (cur_phase == 0) {
                    // acumulando hidden_error: output_delta * wo → producto punto de gradiente → DOT_PRODUCT
                    hidden_error[cur_neuron] += scheduler->mul_heavy(output_delta[cur_k], wo[cur_neuron][cur_k], DOT_PRODUCT);
                    int stall_j = cur_neuron;
                    int stall_k = cur_k;
                    cur_k++;
                    if (cur_k >= OUTPUT_DIM) cur_phase = 1;
                    scheduler->advance_turn();
                    if (!scheduler->check_stall(thread_id, DOT_PRODUCT, /*tag*/ 30u, stall_j, stall_k)) break;

                } else if (cur_phase == 1) {
                    // t*t para derivada de tanh: forma parte del cómputo de activación → ACTIVATION
                    double t = tanh(hidden_input[cur_neuron]);
                    double tt = scheduler->mul_heavy(t, t, ACTIVATION);
                    hidden_delta[cur_neuron] = 1.0 - tt;
                    cur_phase = 2;
                    scheduler->advance_turn();
                    if (!scheduler->check_stall(thread_id, ACTIVATION, /*tag*/ 31u, /*j*/ cur_neuron, /*k*/ 0)) break;

                } else {
                    // hidden_error * deriv = hidden_delta: aplicar derivada de activación → ACTIVATION
                    hidden_delta[cur_neuron] = scheduler->mul_heavy(hidden_error[cur_neuron], hidden_delta[cur_neuron], ACTIVATION);
                    int stall_j = cur_neuron;
                    cur_neuron = -1;
                    scheduler->advance_turn();
                    if (!scheduler->check_stall(thread_id, ACTIVATION, /*tag*/ 32u, /*j*/ stall_j, /*k*/ 0)) break;
                }
            }
        };

        for (int t = 0; t < num_threads; t++) threads.emplace_back(compute_hidden_error, t);
        for (auto& t : threads) t.join();

    // Actualización de pesos hidden -> output
        scheduler->begin_phase();
        threads.clear();
        auto update_wo = [&](int thread_id) {
            int cur_neuron = -1, cur_out = 0;
            bool idle = false;
            while (true) {
                if (!scheduler->wait_turn(thread_id)) break;

                if (idle) {
                    scheduler->advance_turn();

                } else if (cur_neuron < 0) {
                    // FETCH: ciclo NOP de despacho
                    int i = scheduler->take_next(HIDDEN_NEURONS);
                    if (i < 0) {
                        scheduler->mark_done(); idle = true;
                    } else {
                        cur_neuron = i; cur_out = 0;
                    }
                    scheduler->advance_turn();

                } else {
                    // output_delta * hidden_output: gradiente de peso → DOT_PRODUCT
                    double grad = scheduler->mul_heavy(output_delta[cur_out], hidden_output[cur_neuron], DOT_PRODUCT);
                    int stall_j = cur_neuron;
                    int stall_k = cur_out;
                    wo[cur_neuron][cur_out] -= scheduler->mul_total(learning_rate, grad);
                    cur_out++;
                    if (cur_out >= OUTPUT_DIM) cur_neuron = -1;
                    scheduler->advance_turn();
                    if (!scheduler->check_stall(thread_id, DOT_PRODUCT, /*tag*/ 40u, stall_j, stall_k)) break;
                }
            }
        };
        for (int t = 0; t < num_threads; t++) threads.emplace_back(update_wo, t);
        for (auto& t : threads) t.join();
        
    // Bias de salida (fuera de ciclos)
        for (int j = 0; j < OUTPUT_DIM; j++) {
            bo[j] -= scheduler->mul_total(learning_rate, output_delta[j]);
        }
        
    // Actualización de pesos input -> hidden
        scheduler->begin_phase();
        threads.clear();
        auto update_wh = [&](int thread_id) {
            int cur_neuron = -1, cur_dim = 0;
            bool idle = false;
            while (true) {
                if (!scheduler->wait_turn(thread_id)) break;

                if (idle) {
                    scheduler->advance_turn();

                } else if (cur_neuron < 0) {
                    // FETCH: ciclo NOP de despacho
                    int j = scheduler->take_next(HIDDEN_NEURONS);
                    if (j < 0) {
                        scheduler->mark_done(); idle = true;
                    } else {
                        cur_neuron = j; cur_dim = 0;
                    }
                    scheduler->advance_turn();

                } else {
                    // hidden_delta * input: gradiente de peso → DOT_PRODUCT
                    double grad = scheduler->mul_heavy(hidden_delta[cur_neuron], input[cur_dim], DOT_PRODUCT);
                    int stall_j = cur_neuron;
                    int stall_k = cur_dim;
                    wh[cur_dim][cur_neuron] -= scheduler->mul_total(learning_rate, grad);
                    cur_dim++;
                    if (cur_dim >= INPUT_DIM) cur_neuron = -1;
                    scheduler->advance_turn();
                    if (!scheduler->check_stall(thread_id, DOT_PRODUCT, /*tag*/ 50u, stall_j, stall_k)) break;
                }
            }
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