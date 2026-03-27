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

// RoundRobinScheduler modela un procesador fine-grained multihilo:
// cada multiplicación ocupa exactamente un turno del round-robin, luego
// el procesador cambia de hilo (context switch). Esto representa el peor
// caso de granularidad: máximo solapamiento pero máximo overhead de cambio.
class RoundRobinScheduler {
private:
    int num_threads;
    long long context_switch_penalty_cycles;
    long long light_count;   // multiplicaciones forward pass
    long long heavy_count;   // multiplicaciones backward pass
    long long stall_nop_count;
    long long context_switch_count;
    long long total_multiplications;  // operaciones fuera del modelo de ciclos
    long long global_clock;
    int current_turn;
    vector<long long> thread_resume_clock;  // ciclo en que cada hilo puede reanudar tras un stall

    // Latencias en ciclos por tipo de operación:
    //   LIGHT_DOT = 1: w*x en forward pass está pipelined → 1 ciclo.
    //   HEAVY_DOT = 3: w*δ en backward requiere el error previo →
    //                  3 etapas de pipeline antes de que el dato esté listo.
    //   HEAVY_ACT = 2: derivada de tanh (t*t y error*(1-tt)) involucra
    //                  operaciones FP dependientes → 2 ciclos.
    static const long long LATENCY_LIGHT_DOT = 1;
    static const long long LATENCY_HEAVY_DOT = 3;
    static const long long LATENCY_HEAVY_ACT = 2;

    uint32_t stall_seed;

    mutex m;
    condition_variable cv;
    bool phase_open;

    // Hash determinista para simular cache misses de forma reproducible.
    // Combina seed, tag de fase, índices j/k y tipo de operación para que
    // cada multiplicación tenga su propio "dado" fijo entre ejecuciones.
    static inline uint32_t random(uint32_t seed, uint32_t tag, int j, int k, OpType op) {
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

public:
    RoundRobinScheduler(int num_threads_arg, long long context_switch_penalty_cycles_arg = 1,
                        uint32_t stall_seed_arg = 42u)
    : num_threads(num_threads_arg),
    context_switch_penalty_cycles(context_switch_penalty_cycles_arg),
    stall_seed(stall_seed_arg),
    light_count(0), heavy_count(0),
    stall_nop_count(0),
    context_switch_count(0),
    total_multiplications(0), global_clock(0), current_turn(0),
    phase_open(false),
    thread_resume_clock(num_threads_arg, 0LL) {}

    // Abre una nueva fase: habilita a los hilos a competir por turnos.
    // Cada fase corresponde a una etapa del forward/backward pass.
    void begin_phase() {
        unique_lock<mutex> lock(m);
        phase_open = true;
        current_turn = 0;
        fill(thread_resume_clock.begin(), thread_resume_clock.end(), 0LL);
        cv.notify_all();
    }

    // Cierra la fase: los hilos en yield_turns_until_closed salen del bucle.
    void end_phase() {
        unique_lock<mutex> lock(m);
        phase_open = false;
        cv.notify_all();
    }

    bool is_phase_open() {
        lock_guard<mutex> lock(m);
        return phase_open;
    }

    // Bloqueo cooperativo: el hilo espera hasta que sea su turno en el round-robin.
    // Si el hilo sufrió un stall y los otros hilos no cubrieron toda la latencia,
    // avanza el reloj global al mínimo necesario para completarla.
    // Retorna false si la fase ya fue cerrada.
    bool wait_turn(int thread_id) {
        unique_lock<mutex> lock(m);
        cv.wait(lock, [&] { return !phase_open || current_turn == thread_id; });
        if (phase_open && global_clock < thread_resume_clock[thread_id]) {
            global_clock = thread_resume_clock[thread_id];
        }
        return phase_open;
    }

    // Cuando un hilo termina sus neuronas antes que los demás, sigue cediendo
    // turnos para no bloquear el round-robin de los otros hilos.
    // Sin esto, el hilo que termina primero retiene su turno indefinidamente
    // y los demás quedan bloqueados esperando → deadlock.
    void yield_turns_until_closed(int thread_id) {
        while (true) {
            if (!wait_turn(thread_id)) return;
            advance_turn();
        }
    }

    void advance_turn() {
        unique_lock<mutex> lock(m);
        current_turn = (current_turn + 1) % num_threads;
        cv.notify_all();
    }

    // Avanza el turno Y evalúa el stall bajo el mismo lock, evitando la
    // race condition que existiría si advance_turn() y el chequeo del stall
    // fueran dos operaciones separadas.
    //
    // Probabilidades de cache miss (basadas en el perfil de acceso a pesos):
    //   DOT_PRODUCT: 2.4% de miss → stall de 8 ciclos (pesos del gradiente,
    //                acceso no secuencial en memoria)
    //   ACTIVATION:  0.6% de miss → stall de 3 ciclos (valores intermedios,
    //                mejor localidad porque son del mismo array hidden_input)
    //
    // Retorna false si la fase se cerró.
    bool advance_and_check_stall(int thread_id, OpType op, uint32_t tag, int j, int k) {
        unique_lock<mutex> lock(m);
        current_turn = (current_turn + 1) % num_threads;
        cv.notify_all();
        if (!phase_open) return false;
        uint32_t miss_pct1000 = (op == DOT_PRODUCT) ? 24u : 6u;
        long long stall_lat   = (op == DOT_PRODUCT) ? 8LL : 3LL;
        uint32_t pct = random(stall_seed, tag, j, k, op) % 1000u;
        if (pct < miss_pct1000) {
            stall_nop_count++;
            thread_resume_clock[thread_id] = global_clock + stall_lat;
        }
        return true;
    }

private:
    // Selecciona la latencia de context switch según tipo y peso de la operación.
    long long pick_latency(bool is_heavy, OpType op) const {
        if (!is_heavy) {
            return LATENCY_LIGHT_DOT;
        } else {
            return (op == ACTIVATION) ? LATENCY_HEAVY_ACT : LATENCY_HEAVY_DOT;
        }
    }

public:
    // mul_light: multiplicación de forward pass (w*x).
    // Cuesta 1 ciclo base + latencia de context switch en modo multihilo.
    template <typename T>
    T mul_light(T a, T b, OpType op) {
        lock_guard<mutex> lock(m);
        light_count++;
        global_clock++;
        if (num_threads > 1) {
            context_switch_count++;
            global_clock += pick_latency(false, op);
        }
        return a * b;
    }

    // mul_heavy: multiplicación de backward pass (w*δ, t*t, error*(1-tt)).
    // Cuesta más ciclos que light porque los gradientes tienen mayor profundidad
    // de pipeline y peor localidad de caché.
    template <typename T>
    T mul_heavy(T a, T b, OpType op) {
        lock_guard<mutex> lock(m);
        heavy_count++;
        global_clock++;
        if (num_threads > 1) {
            context_switch_count++;
            global_clock += pick_latency(true, op);
        }
        return a * b;
    }

    // mul_total: operaciones fuera del modelo de ciclos del round-robin
    // (lr*grad, error², etc.). Se contabilizan para métricas globales pero
    // no incrementan global_clock porque ocurren fuera de la fase simulada.
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
    int       get_num_threads()           const { return num_threads; }
};


// Red neuronal de una capa oculta con scheduler fine-grained.
// Las capas se paralelizan por neuronas con scheduling estático (strided):
// el hilo t procesa las neuronas t, t+T, t+2T, ...
// El scheduler coordina el orden de ejecución y acumula las métricas de
// ciclos, stalls y context switches para el CSV de resultados.
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

    std::mt19937 rng;

    double uniform_01() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(rng);
    }

public:
    // Inicialización de Xavier: escala los pesos según la dimensión de entrada
    // para mantener la varianza de activaciones estable entre capas.
    NeuralNetworkFineGrained(RoundRobinScheduler* sched, uint32_t nn_seed = 42u) : scheduler(sched) {
        rng.seed(nn_seed);

        for (int i = 0; i < INPUT_DIM; i++)
            for (int j = 0; j < HIDDEN_NEURONS; j++)
                wh[i][j] = (uniform_01() - 0.5) * sqrt(2.0 / INPUT_DIM);

        for (int j = 0; j < HIDDEN_NEURONS; j++)
            bh[j] = 0.0;

        for (int i = 0; i < HIDDEN_NEURONS; i++)
            for (int j = 0; j < OUTPUT_DIM; j++)
                wo[i][j] = (uniform_01() - 0.5) * sqrt(2.0 / HIDDEN_NEURONS);

        for (int j = 0; j < OUTPUT_DIM; j++)
            bo[j] = 0.0;
    }

    // Forward pass en dos fases separadas por una barrera implícita (join).
    // Fase 1 (hidden): cada hilo calcula sus neuronas ocultas (w*x + b → tanh).
    // Fase 2 (output): cada hilo acumula su parte de la suma ponderada de hidden.
    // Las fases no pueden solaparse porque la capa de salida necesita todos los
    // hidden_output[] completos. El atómico *_finished cuenta cuántos hilos
    // terminaron para que el último llame a end_phase() sin coordinador externo.
    double forward_finegrained(const double input[]) {
        vector<thread> threads;
        int num_threads = scheduler->get_num_threads();

        // --- Fase 1: capa oculta ---
        scheduler->begin_phase();
        std::atomic<int> hidden_finished{0};
        auto compute_hidden = [&](int thread_id) {
            for (int j = thread_id; j < HIDDEN_NEURONS; j += num_threads) {
                hidden_input[j] = bh[j];
                for (int k = 0; k < INPUT_DIM; k++) {
                    if (!scheduler->wait_turn(thread_id)) return;
                    hidden_input[j] += scheduler->mul_light(input[k], wh[k][j], DOT_PRODUCT);
                    if (!scheduler->advance_and_check_stall(thread_id, DOT_PRODUCT, 10u, j, k)) return;
                }
                hidden_output[j] = tanh_activation(hidden_input[j]);
            }
            if (hidden_finished.fetch_add(1) + 1 == num_threads) {
                scheduler->end_phase();
            }
            scheduler->yield_turns_until_closed(thread_id);
        };

        for (int t = 0; t < num_threads; t++) threads.emplace_back(compute_hidden, t);
        for (auto& t : threads) t.join();

        // --- Fase 2: capa de salida ---
        scheduler->begin_phase();
        std::atomic<int> output_finished{0};
        for (int outj = 0; outj < OUTPUT_DIM; outj++)
            output_input[outj] = bo[outj];

        auto compute_output = [&](int thread_id) {
            for (int i = thread_id; i < HIDDEN_NEURONS; i += num_threads) {
                for (int outj = 0; outj < OUTPUT_DIM; outj++) {
                    if (!scheduler->wait_turn(thread_id)) return;
                    output_input[outj] += scheduler->mul_light(hidden_output[i], wo[i][outj], DOT_PRODUCT);
                    if (!scheduler->advance_and_check_stall(thread_id, DOT_PRODUCT, 20u, i, outj)) return;
                }
            }
            if (output_finished.fetch_add(1) + 1 == num_threads) {
                scheduler->end_phase();
            }
            scheduler->yield_turns_until_closed(thread_id);
        };

        threads.clear();
        for (int t = 0; t < num_threads; t++) threads.emplace_back(compute_output, t);
        for (auto& t : threads) t.join();

        for (int outj = 0; outj < OUTPUT_DIM; outj++)
            output[outj] = output_input[outj];

        return output[0];
    }

    // Backward pass en cuatro fases. Las tres primeras son paralelas con scheduler;
    // la cuarta (bias) es secuencial fuera de ciclos.
    //
    // Fase 1: hidden_error = woᵀ * output_delta; hidden_delta = error * tanh'(x)
    // Fase 2: wo -= lr * output_delta * hidden_outputᵀ
    // Fase 3: wh -= lr * hidden_delta * inputᵀ
    // Fase 4: actualizar biases (mul_total, no consume ciclos del scheduler)
    void backward_finegrained(const double input[], double target, double learning_rate) {

        // Gradiente de MSE: dL/dy = y - target
        double output_delta[OUTPUT_DIM];
        for (int j = 0; j < OUTPUT_DIM; j++)
            output_delta[j] = output[j] - target;

        double hidden_error[HIDDEN_NEURONS];
        double hidden_delta[HIDDEN_NEURONS];

        int num_threads = scheduler->get_num_threads();
        vector<thread> threads;

        // --- Fase 1: error y delta de la capa oculta ---
        scheduler->begin_phase();
        std::atomic<int> bhe_finished{0};
        auto compute_hidden_error = [&](int thread_id) {
            for (int j = thread_id; j < HIDDEN_NEURONS; j += num_threads) {
                hidden_error[j] = 0.0;
                for (int k = 0; k < OUTPUT_DIM; k++) {
                    if (!scheduler->wait_turn(thread_id)) return;
                    hidden_error[j] += scheduler->mul_heavy(output_delta[k], wo[j][k], DOT_PRODUCT);
                    if (!scheduler->advance_and_check_stall(thread_id, DOT_PRODUCT, 30u, j, k)) return;
                }

                // tanh'(x) = 1 - tanh²(x): se modela con dos mul_heavy porque
                // t*t y error*(1-tt) son ops FP dependientes con mayor latencia
                // que las del forward pass (datos intermedios, peor localidad).
                if (!scheduler->wait_turn(thread_id)) return;
                double t = tanh(hidden_input[j]);
                double tt = scheduler->mul_heavy(t, t, ACTIVATION);
                if (!scheduler->advance_and_check_stall(thread_id, ACTIVATION, 31u, j, 0)) return;

                if (!scheduler->wait_turn(thread_id)) return;
                hidden_delta[j] = scheduler->mul_heavy(hidden_error[j], (1.0 - tt), ACTIVATION);
                if (!scheduler->advance_and_check_stall(thread_id, ACTIVATION, 32u, j, 0)) return;
            }
            if (bhe_finished.fetch_add(1) + 1 == num_threads) scheduler->end_phase();
            scheduler->yield_turns_until_closed(thread_id);
        };

        for (int t = 0; t < num_threads; t++) threads.emplace_back(compute_hidden_error, t);
        for (auto& t : threads) t.join();

        // --- Fase 2: actualizar pesos hidden → output ---
        scheduler->begin_phase();
        std::atomic<int> uwo_finished{0};
        threads.clear();
        auto update_wo = [&](int thread_id) {
            for (int i = thread_id; i < HIDDEN_NEURONS; i += num_threads) {
                for (int outj = 0; outj < OUTPUT_DIM; outj++) {
                    if (!scheduler->wait_turn(thread_id)) return;
                    double grad = scheduler->mul_heavy(output_delta[outj], hidden_output[i], DOT_PRODUCT);
                    wo[i][outj] -= scheduler->mul_total(learning_rate, grad);
                    if (!scheduler->advance_and_check_stall(thread_id, DOT_PRODUCT, 40u, i, outj)) return;
                }
            }
            if (uwo_finished.fetch_add(1) + 1 == num_threads) scheduler->end_phase();
            scheduler->yield_turns_until_closed(thread_id);
        };
        for (int t = 0; t < num_threads; t++) threads.emplace_back(update_wo, t);
        for (auto& t : threads) t.join();

        // Bias de salida fuera de ciclos (escalar, no justifica paralelismo)
        for (int j = 0; j < OUTPUT_DIM; j++)
            bo[j] -= scheduler->mul_total(learning_rate, output_delta[j]);

        // --- Fase 3: actualizar pesos input → hidden ---
        scheduler->begin_phase();
        std::atomic<int> uwh_finished{0};
        threads.clear();
        auto update_wh = [&](int thread_id) {
            for (int j = thread_id; j < HIDDEN_NEURONS; j += num_threads) {
                for (int k = 0; k < INPUT_DIM; k++) {
                    if (!scheduler->wait_turn(thread_id)) return;
                    double grad = scheduler->mul_heavy(hidden_delta[j], input[k], DOT_PRODUCT);
                    wh[k][j] -= scheduler->mul_total(learning_rate, grad);
                    if (!scheduler->advance_and_check_stall(thread_id, DOT_PRODUCT, 50u, j, k)) return;
                }
            }
            if (uwh_finished.fetch_add(1) + 1 == num_threads) scheduler->end_phase();
            scheduler->yield_turns_until_closed(thread_id);
        };
        for (int t = 0; t < num_threads; t++) threads.emplace_back(update_wh, t);
        for (auto& t : threads) t.join();

        // Bias de capa oculta fuera de ciclos
        for (int j = 0; j < HIDDEN_NEURONS; j++)
            bh[j] -= scheduler->mul_total(learning_rate, hidden_delta[j]);
    }

    void train(const vector<vector<double>>& X, const vector<double>& Y,
               int epochs, double learning_rate) {
        int n_samples = X.size();

        for (int epoch = 0; epoch < epochs; epoch++) {
            double total_loss = 0.0;

            for (int i = 0; i < n_samples; i++) {
                double input[INPUT_DIM];
                for (int j = 0; j < INPUT_DIM; j++)
                    input[j] = X[i][j];

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
            for (int j = 0; j < INPUT_DIM; j++)
                input[j] = X[i][j];

            double prediction = forward_finegrained(input);
            double error = prediction - Y[i];
            total_loss += scheduler->mul_total(error, error);
        }

        return total_loss / n_samples;
    }

    RoundRobinScheduler* get_scheduler() { return scheduler; }
};
