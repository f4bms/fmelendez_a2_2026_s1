// Algoritmo de entrenamiento Round-Robin (fine-grained multithreading - FGMT) con hilos.
// - El "context switch" (cambio de turno) ocurre en CADA multiplicación (1 multiplicación = 1 ciclo).
// - Para mantener consistencia con CGMT, se clasifica cada multiplicación como:
//   * liviana (light): operaciones del forward.
//   * pesada (heavy): operaciones del backprop y actualización de pesos.
// - Operaciones "fuera de ciclos" (lr*grad, error^2, etc.) se contabilizan aparte con mul_total().

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <chrono>
#include <iomanip>
#include <random>

using namespace std;
using namespace std::chrono;

const int INPUT_DIM = 3;
const int HIDDEN_NEURONS = 30;
const int OUTPUT_DIM = 1;
const double LEARNING_RATE = 0.08;
const int EPOCHS = 1500;

double tanh_activation(double x) {
    return tanh(x);
}

double tanh_derivative(double x) {
    double t = tanh(x);
    return 1.0 - t * t;
}

double basefunction(double x[], int d){
    double sum = 0;
    for(int i=0;i<d;i++){
        sum += sin(x[i]) + 0.3*x[i]*x[i];
    }
    return sum;
}

// Scheduler Round-Robin (FGMT): un único hilo tiene el turno, y luego se avanza al siguiente.
class RoundRobinScheduler {
private:
    int num_threads;
    long long light_count;
    long long heavy_count;
    long long stall_nop_count;
    long long total_multiplications;
    long long global_clock;
    int current_turn;

    mutex m;
    condition_variable cv;
    int phase_done_count;
    bool phase_open;

    atomic<int> next_neuron;
    static const int STALL_EVERY = 10000;
    vector<int> thread_mul_counter;

public:
    RoundRobinScheduler(int num_threads_arg, long long /*quantum_size*/ = 1)
    : num_threads(num_threads_arg),
      light_count(0), heavy_count(0), stall_nop_count(0),
      total_multiplications(0), global_clock(0), current_turn(0),
      phase_done_count(0), phase_open(false),
      next_neuron(0),
      thread_mul_counter(num_threads_arg, 0) {}

    void begin_phase() {
        next_neuron.store(0, memory_order_relaxed);
        unique_lock<mutex> lock(m);
        phase_open = true;
        phase_done_count = 0;
        current_turn = 0;
        cv.notify_all();
    }

    // Espera el turno del hilo. Retorna true si la fase sigue abierta.
    // Es el ÚNICO punto de bloqueo: no se llama desde ninguna otra función.
    bool wait_turn(int thread_id) {
        unique_lock<mutex> lock(m);
        cv.wait(lock, [&] { return !phase_open || current_turn == thread_id; });
        return phase_open;
    }

    // Avanza el turno. Siempre se llama después de wait_turn.
    void advance_turn() {
        unique_lock<mutex> lock(m);
        global_clock++;
        current_turn = (current_turn + 1) % num_threads;
        cv.notify_all();
    }

    // Registra que este hilo terminó su trabajo.
    // NO participa en la rotación — solo cierra la fase cuando todos terminaron.
    void mark_done() {
        unique_lock<mutex> lock(m);
        if (!phase_open) return;
        phase_done_count++;
        if (phase_done_count >= num_threads) phase_open = false;
        cv.notify_all();
    }

    // Toma el siguiente índice de neurona disponible. Retorna -1 si no hay más.
    int take_next(int total) {
        int idx = next_neuron.fetch_add(1, memory_order_relaxed);
        return (idx < total) ? idx : -1;
    }

    // Contadores de multiplicaciones (llamar dentro del turno, con la fase garantizada abierta).
    // Consistencia con CGMT: forward => count_light(); backprop/update => count_heavy().
    void count_light() { lock_guard<mutex> l(m); light_count++; }
    void count_heavy() { lock_guard<mutex> l(m); heavy_count++; }

    // Multiplicaciones EN ciclos.
    // En FGMT, cada llamada debe ejecutarse DENTRO del turno del hilo (entre wait_turn y advance_turn).
    template <typename T>
    T mul_light(T a, T b) {
        count_light();
        return a * b;
    }

    template <typename T>
    T mul_heavy(T a, T b) {
        count_heavy();
        return a * b;
    }

    // Stall: si el hilo llegó a STALL_EVERY multiplicaciones, consume UN turno adicional como NOP.
    // Retorna false si la fase cerró durante el stall (el hilo debe salir).
    bool check_stall(int thread_id) {
        thread_mul_counter[thread_id]++;
        if (thread_mul_counter[thread_id] >= STALL_EVERY) {
            thread_mul_counter[thread_id] = 0;
            if (!wait_turn(thread_id)) return false; // fase cerró → salir
            { lock_guard<mutex> l(m); stall_nop_count++; }
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

    // --- getters ---
    long long get_light_count()           const { return light_count; }
    long long get_heavy_count()           const { return heavy_count; }
    long long get_stall_nop_count()       const { return stall_nop_count; }
    long long get_total_multiplications() const { return total_multiplications; }
    long long get_global_clock()          const { return global_clock; }
    int       get_num_threads()           const { return num_threads; }

    void print_stats() {
        long long cycle_muls = light_count + heavy_count;
        cout << "\n --- metricas ---" << endl;
        cout << "multiplicaciones light (ciclos):  " << light_count          << endl;
        cout << "multiplicaciones heavy (ciclos):  " << heavy_count          << endl;
        cout << "total multiplicaciones en ciclos: " << cycle_muls           << endl;
        cout << "multiplicaciones fuera de ciclos: " << total_multiplications << endl;
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

    // RNG por instancia (evita rand()/srand() globales que no son thread-safe)
    std::mt19937 rng;

    double uniform_symmetric(double scale) {
        // [-scale, scale]
        std::uniform_real_distribution<double> dist(-scale, scale);
        return dist(rng);
    }
    
public:
    NeuralNetworkFineGrained(RoundRobinScheduler* sched) : scheduler(sched) {
        // Semilla por red, evitando estado global compartido.
        // random_device puede ser lento/no determinista; se mezcla con reloj para robustez.
        std::random_device rd;
        auto now = static_cast<unsigned>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::seed_seq seed{rd(), now, static_cast<unsigned>(reinterpret_cast<uintptr_t>(this))};
        rng.seed(seed);
        
        // Inicializar pesos input -> hidden
        for (int i = 0; i < INPUT_DIM; i++) {
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                // equivalente a (rand/RAND_MAX - 0.5) * sqrt(2/input)
                wh[i][j] = uniform_symmetric(0.5) * sqrt(2.0 / INPUT_DIM);
            }
        }
        
        // Inicializar bias hidden
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            bh[j] = 0.0;
        }
        
        // Inicializar pesos hidden -> output
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            for (int j = 0; j < OUTPUT_DIM; j++) {
                wo[i][j] = uniform_symmetric(0.5) * sqrt(2.0 / HIDDEN_NEURONS);
            }
        }
        
        // Inicializar bias output
        for (int j = 0; j < OUTPUT_DIM; j++) {
            bo[j] = 0.0;
        }
    }
    
    // Forward propagation ------------------------------------------------------
    double forward_finegrained(const double input[]) {
        vector<thread> threads;
        int num_threads = scheduler->get_num_threads();

    /* Modelo de turno único para todos los lambdas:
       Cada iteración del while es exactamente: wait_turn → acción → advance_turn
       Sin wait_turns anidados, sin muls con wait interno.
       Estado por hilo: cur_neuron (-1 = necesita fetch), cur_dim, idle (sin más trabajo) */

    scheduler->begin_phase();
        auto compute_hidden = [&](int thread_id) {
            int cur_neuron = -1, cur_dim = 0;
            bool idle = false;
            while (true) {
                if (!scheduler->wait_turn(thread_id)) break; // fase cerrada

                if (idle) {
                    // NOP: mantiene la rotación hasta que todos terminen
                    scheduler->advance_turn();

                } else if (cur_neuron < 0) {
                    // FETCH: tomar neurona — este ciclo es un NOP de despacho,
                    // no se hace ninguna multiplicación. La primera mul ocurre
                    // en el siguiente ciclo (bloque WORK abajo).
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
                    // WORK: una multiplicación por ciclo — round-robin puro
                    // Forward => multiplicación liviana
                    hidden_input[cur_neuron] += scheduler->mul_light(input[cur_dim], wh[cur_dim][cur_neuron]);
                    cur_dim++;
                    if (cur_dim >= INPUT_DIM) {
                        hidden_output[cur_neuron] = tanh_activation(hidden_input[cur_neuron]);
                        cur_neuron = -1;
                    }
                    scheduler->advance_turn();
                    if (!scheduler->check_stall(thread_id)) break;
                }
            }
        };

        //terminación de la fase oculta
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
                    // FETCH: ciclo NOP de despacho
                    int i = scheduler->take_next(HIDDEN_NEURONS);
                    if (i < 0) {
                        scheduler->mark_done(); idle = true;
                    } else {
                        cur_neuron = i; cur_out = 0;
                    }
                    scheduler->advance_turn();

                } else {
                    // WORK: una multiplicación por ciclo
                    // Forward => multiplicación liviana
                    output_input[cur_out] += scheduler->mul_light(hidden_output[cur_neuron], wo[cur_neuron][cur_out]);
                    cur_out++;
                    if (cur_out >= OUTPUT_DIM) cur_neuron = -1;
                    scheduler->advance_turn();
                    if (!scheduler->check_stall(thread_id)) break;
                }
            }
        };

// Finalizacion de la capa de salida
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
        
        // Gradiente de MSE respetcto a la salida
        double output_delta[OUTPUT_DIM];
        for (int j = 0; j < OUTPUT_DIM; j++) {
            output_delta[j] = output[j] - target;
        }
        
        double hidden_error[HIDDEN_NEURONS];
        double hidden_delta[HIDDEN_NEURONS];

    // hidden_error/hidden_delta (equivalente a CGMT: mul_cgmt_heavy)
        int num_threads = scheduler->get_num_threads();
        scheduler->begin_phase();
        vector<thread> threads;
        auto compute_hidden_error = [&](int thread_id) {
            // Estado: qué neurona proceso y en qué mul estoy
            // Cada neurona backward hace:
            // - OUTPUT_DIM multiplicaciones para acumular hidden_error.
            // - 1 multiplicación para t*t (derivada de tanh).
            // - 1 multiplicación para hidden_error * deriv (hidden_delta).
            // En CGMT, TODAS estas multiplicaciones se consideran "pesadas".
            // Las agrupamos en fases: fase 0=error_acum, fase 1=t*t, fase 2=delta.
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
                    // acumulando hidden_error — una mul por ciclo
                    hidden_error[cur_neuron] += scheduler->mul_heavy(output_delta[cur_k], wo[cur_neuron][cur_k]);
                    cur_k++;
                    if (cur_k >= OUTPUT_DIM) cur_phase = 1;
                    scheduler->advance_turn();
                    if (!scheduler->check_stall(thread_id)) break;

                } else if (cur_phase == 1) {
                    // mul t*t para la derivada de tanh
                    double t = tanh(hidden_input[cur_neuron]);
                    double tt = scheduler->mul_heavy(t, t);
                    hidden_delta[cur_neuron] = 1.0 - tt; // guardamos deriv temporalmente
                    cur_phase = 2;
                    scheduler->advance_turn();
                    if (!scheduler->check_stall(thread_id)) break;

                } else {
                    // mul hidden_error * deriv = hidden_delta
                    hidden_delta[cur_neuron] = scheduler->mul_heavy(hidden_error[cur_neuron], hidden_delta[cur_neuron]);
                    cur_neuron = -1;
                    scheduler->advance_turn();
                    if (!scheduler->check_stall(thread_id)) break;
                }
            }
        };

        for (int t = 0; t < num_threads; t++) threads.emplace_back(compute_hidden_error, t);
        for (auto& t : threads) t.join();

    // Actualización de pesos hidden -> output (equivalente a CGMT: mul_cgmt_heavy)
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
                    // WORK: una mul por ciclo — backward heavy
                    double grad = scheduler->mul_heavy(output_delta[cur_out], hidden_output[cur_neuron]);
                    wo[cur_neuron][cur_out] -= scheduler->mul_total(learning_rate, grad);
                    cur_out++;
                    if (cur_out >= OUTPUT_DIM) cur_neuron = -1;
                    scheduler->advance_turn();
                    if (!scheduler->check_stall(thread_id)) break;
                }
            }
        };
        for (int t = 0; t < num_threads; t++) threads.emplace_back(update_wo, t);
        for (auto& t : threads) t.join();
        
    // Bias de salida (fuera de ciclos, igual que CGMT)
        for (int j = 0; j < OUTPUT_DIM; j++) {
            // lr * delta (total)
            bo[j] -= scheduler->mul_total(learning_rate, output_delta[j]);
        }
        
    // Actualización de pesos input -> hidden (equivalente a CGMT: mul_cgmt_heavy)
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
                    // WORK: una mul por ciclo — backward heavy
                    double grad = scheduler->mul_heavy(hidden_delta[cur_neuron], input[cur_dim]);
                    wh[cur_dim][cur_neuron] -= scheduler->mul_total(learning_rate, grad);
                    cur_dim++;
                    if (cur_dim >= INPUT_DIM) cur_neuron = -1;
                    scheduler->advance_turn();
                    if (!scheduler->check_stall(thread_id)) break;
                }
            }
        };
        for (int t = 0; t < num_threads; t++) threads.emplace_back(update_wh, t);
        for (auto& t : threads) t.join();
        
    // Bias de capa oculta (fuera de ciclos, igual que CGMT)
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

//cargar datos del archivo txt
void loadDataset(const string& filename, vector<vector<double>>& X, vector<double>& Y) {
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: No se pudo abrir el archivo" << endl;
        return;
    }
    
    double x1, x2, x3, y;
    while (file >> x1 >> x2 >> x3 >> y) {
        X.push_back({x1, x2, x3});
        Y.push_back(y);
    }
    
    file.close();
    cout << "Dataset cargado: " << X.size() << " muestras" << endl;
}