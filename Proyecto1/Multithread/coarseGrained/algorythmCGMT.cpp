// Algoritmo de entrenamiento (coarse-grained multithreading - CGMT) con hilos.
// - Solo hay 1 hilo "activo" a la vez (los demás esperan).
// - La ÚNICA diferencia con FGMT debe ser el scheduler:
//     * FGMT: round-robin por multiplicación.
//     * CGMT: el hilo activo solo cambia cuando ocurre un stall.
// - Por lo tanto, aquí:
//     * Stalls: mismos criterios que FGMT (probabilidad por OpType, determinista por-evento).
//     * Context switch: ocurre SIEMPRE que hay stall y existe otro hilo disponible.
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <iomanip>
#include <random>
#include <atomic>

#include "../../common/nn_utils.h"
#include "../../common/dataset.h"

using namespace std;
using namespace std::chrono;

// (constantes/funciones comunes están en common/)

// Scheduler CGMT:
// - Hay 1 hilo "activo" a la vez.
// - El hilo activo ejecuta multiplicaciones en "ciclos" (global_clock++).
// - Un stall ocurre cada N multiplicaciones "pesadas" del hilo activo, donde N es aleatorio
//   en un rango configurable (por hilo).
// - Al ocurrir un stall se agrega una latencia (stall_latency_cycles) y, típicamente, se hace
//   un context switch (con una penalidad adicional). Con cierta probabilidad, el switch se omite.
class CoarseGrainedScheduler {
private:
	int num_threads;
	long long stall_latency_cycles; // cuántos ciclos dura el stall (sin computar)
	long long context_switch_penalty_cycles;

	// Latencias alineadas con FGMT (mismas constantes/criterios).
	static const long long LATENCY_LIGHT_DOT = 1;
	static const long long LATENCY_LIGHT_ACT = 1;
	static const long long LATENCY_HEAVY_DOT = 3;
	static const long long LATENCY_HEAVY_ACT = 2;

	// Métricas
	long long cycle_mul_light_count;
	long long cycle_mul_heavy_count;
	long long uncounted_mul_count;
	long long compute_multiplications; // (legacy) livianas + pesadas
	long long total_multiplications;
	long long stall_count;
	long long stall_events_on_heavy_count;
	long long context_switch_count;
	long long fetch_count;
	long long nop_count;
	long long global_clock;

	// Estado
	int current_active_thread;
	bool phase_open;
	int phase_done_count;
	vector<bool> thread_done;

	// Asignación de trabajo por cola (igual que FGMT): cada hilo toma el siguiente índice.
	std::atomic<int> next_index;
	// Nota: en la versión anterior había un PRNG por hilo y un umbral aleatorio
	// para disparar stalls. Ahora usamos el mismo criterio que FGMT:
	// probabilístico por OpType, determinista por evento (tag,j,k).

	mutex m;
	condition_variable cv;

	// Pseudo-random determinista (stateless), igual a FGMT, para reproducir el
	// mismo patrón de stalls por evento sin depender del orden de ejecución.
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

	long long pick_latency(bool is_heavy, OpType op) const {
		if (!is_heavy) {
			return (op == ACTIVATION) ? LATENCY_LIGHT_ACT : LATENCY_LIGHT_DOT;
		} else {
			return (op == ACTIVATION) ? LATENCY_HEAVY_ACT : LATENCY_HEAVY_DOT;
		}
	}

	void switch_to_next_thread_locked() {
		int next = (current_active_thread + 1) % num_threads;
		int spins = 0;
		while (spins < num_threads && thread_done[next]) {
			next = (next + 1) % num_threads;
			spins++;
		}
		if (!thread_done[next] && next != current_active_thread) {
			current_active_thread = next;
		}
	}

	long long unfinished_threads_locked() const {
		long long u = 0;
		for (bool done : thread_done) {
			if (!done) u++;
		}
		return u;
	}

public:
	CoarseGrainedScheduler(int num_threads_arg,
						   long long stall_latency_cycles_arg = 1,
						   long long context_switch_penalty_cycles_arg = 1,
						   unsigned int /*fixed_seed*/ = 42u)
		: num_threads(num_threads_arg), stall_latency_cycles(stall_latency_cycles_arg),
		  context_switch_penalty_cycles(context_switch_penalty_cycles_arg),
		  cycle_mul_light_count(0), cycle_mul_heavy_count(0), uncounted_mul_count(0),
		  compute_multiplications(0), total_multiplications(0),
		  stall_count(0), stall_events_on_heavy_count(0), context_switch_count(0), fetch_count(0), nop_count(0),
		  global_clock(0), current_active_thread(0), phase_open(false),
		  phase_done_count(0), thread_done(num_threads_arg, false), next_index(0) {
	}

	void begin_phase() {
		unique_lock<mutex> lock(m);
		next_index.store(0, std::memory_order_relaxed);
		phase_open = true;
		phase_done_count = 0;
		current_active_thread = 0;
		fill(thread_done.begin(), thread_done.end(), false);
		cv.notify_all();
	}

	// Toma el siguiente índice de trabajo disponible. Retorna -1 si ya no hay más.
	int take_next(int total) {
		// Simula el costo de hacer fetch de la cola (1 ciclo).
		{
			lock_guard<mutex> lock(m);
			fetch_count++;
			global_clock += 1;
		}
		int idx = next_index.fetch_add(1, std::memory_order_relaxed);
		return (idx < total) ? idx : -1;
	}

	void mark_done(int thread_id) {
		unique_lock<mutex> lock(m);
		if (!phase_open) return;
		if (!thread_done[thread_id]) {
			thread_done[thread_id] = true;
			phase_done_count++;

			if (current_active_thread == thread_id && phase_done_count < num_threads) {
				switch_to_next_thread_locked();
			}

			if (phase_done_count >= num_threads) {
				phase_open = false;
			}
		}
		cv.notify_all();
	}

	bool is_phase_open() {
		lock_guard<mutex> lock(m);
		return phase_open;
	}

	int get_num_threads() const { return num_threads; }

	// Getters de métricas (para logging/CSV)
	long long get_light_count() const { return cycle_mul_light_count; }
	long long get_heavy_count() const { return cycle_mul_heavy_count; }
	long long get_total_multiplications() const { return total_multiplications; }
	long long get_stall_count() const { return stall_count; }
	long long get_context_switch_count() const { return context_switch_count; }
	long long get_global_clock() const { return global_clock; }
	long long get_fetch_count() const { return fetch_count; }

	void wait_active(int thread_id) {
		unique_lock<mutex> lock(m);
		cv.wait(lock, [&] { return !phase_open || current_active_thread == thread_id; });
	}

	// 1) Multiplicación normal: NO se contabiliza y NO consume ciclos del scheduler.
	template <typename T>
	T mul_normal(T a, T b) {
		return a * b;
	}

	// 1b) Multiplicación NO contada como ciclo, pero SÍ registrada (útil para métricas).
	template <typename T>
	T mul_uncounted(T a, T b) {
		uncounted_mul_count++;
		total_multiplications++;
		return a * b;
	}

	// Helper interno: cuerpo común de la multiplicación CGMT.
	template <typename T>
	T mul_cgmt_impl(int thread_id, T a, T b, bool is_heavy, OpType op, uint32_t tag, int j, int k) {
		while (true) {
			wait_active(thread_id);
			if (!is_phase_open()) return T{};

			unique_lock<mutex> lock(m);
			if (!phase_open) return T{};
			if (current_active_thread != thread_id) continue;

			// Ejecutar 1 multiplicación del hilo activo (cuenta como 1 ciclo de cómputo).
			compute_multiplications++;
			total_multiplications++;
			global_clock++;
			T r = a * b;

			// Contabilizar tipo de multiplicación.
		if (is_heavy) {
			cycle_mul_heavy_count++;
		} else {
			cycle_mul_light_count++;
		}

			// CGMT NO cambia de hilo salvo que ocurra stall.
			// Mismo criterio probabilístico por operación que FGMT + determinismo por evento.
			double base_miss_rate = 2.0; // % base
			double pressure = (op == DOT_PRODUCT) ? 1.2 : 0.3;
			double miss_prob = base_miss_rate * pressure; // % final
			static constexpr uint32_t SEED = 42u;
			uint32_t pct = random(SEED, tag, j, k, op) % 100u;
			if (pct < static_cast<uint32_t>(miss_prob)) {
				stall_count++;
				if (is_heavy) stall_events_on_heavy_count++;

				// Costo del stall: igual que FGMT (penalidad fija = 1 ciclo),
				// más la latencia configurable si se quiere modelar adicional.
				global_clock += 1;
				if (stall_latency_cycles > 0) {
					global_clock += stall_latency_cycles;
				}

				// Al haber stall, este scheduler hace context switch SIEMPRE (si hay otro hilo vivo).
				int old = current_active_thread;
				if (unfinished_threads_locked() > 1) {
					switch_to_next_thread_locked();
					if (current_active_thread != old) {
						context_switch_count++;
						global_clock += pick_latency(is_heavy, op);
						if (context_switch_penalty_cycles > 0) {
							global_clock += context_switch_penalty_cycles;
						}
					}
				}
				cv.notify_all();
			}

			return r;
		}
	}

	// 2) Multiplicación liviana: se cuenta como ciclo (NO cambia de hilo salvo stall).
	template <typename T>
	T mul_cgmt_light(int thread_id, T a, T b, OpType op, uint32_t tag, int j, int k) {
		return mul_cgmt_impl(thread_id, a, b, false, op, tag, j, k);
	}

	// 3) Multiplicación pesada: se cuenta como ciclo y puede provocar stalls (y, por ende, switches).
	template <typename T>
	T mul_cgmt_heavy(int thread_id, T a, T b, OpType op, uint32_t tag, int j, int k) {
		return mul_cgmt_impl(thread_id, a, b, true, op, tag, j, k);
	}

	template <typename T>
	T mul_cgmt(int thread_id, T a, T b) {
		// Compatibilidad: por defecto se considera liviana.
			return mul_cgmt_light(thread_id, a, b, DOT_PRODUCT, 0u, 0, 0);
	}

	template <typename T>
	T mul_total(T a, T b) {
		return mul_uncounted(a, b);
	}

	void idle_cgmt(int thread_id) {
		wait_active(thread_id);
		if (!is_phase_open()) return;
		unique_lock<mutex> lock(m);
		if (!phase_open || current_active_thread != thread_id) return;
		nop_count++;
		global_clock++;
		cv.notify_all();
	}

	void print_stats() {
		cout << "\n --- metricas CGMT ---" << endl;
		cout << "multiplicaciones/ciclos (livianas+pesadas): " << compute_multiplications << endl;
		cout << "  - livianas/ciclos: " << cycle_mul_light_count << endl;
		cout << "  - pesadas/ciclos: " << cycle_mul_heavy_count << endl;
		cout << "fetches (take_next): " << fetch_count << endl;
		cout << "multiplicaciones no contadas (registradas): " << uncounted_mul_count << endl;
		cout << "multiplicaciones totales (registradas): " << total_multiplications << endl;
		cout << "stalls (prob por OpType, determinista por evento): " << stall_count << endl;
		cout << "context switch (la mayoria en stalls): " << context_switch_count << endl;
		cout << "NOPs: " << nop_count << endl;
		cout << "ciclos simulados: " << global_clock << endl;
		cout << "-----------------------\n" << endl;
	}
};

class NeuralNetworkCoarseGrained {
private:
	double wh[INPUT_DIM][HIDDEN_NEURONS];
	double bh[HIDDEN_NEURONS];
	double wo[HIDDEN_NEURONS][OUTPUT_DIM];
	double bo[OUTPUT_DIM];

	double hidden_input[HIDDEN_NEURONS];
	double hidden_output[HIDDEN_NEURONS];
	double output_input[OUTPUT_DIM];
	double output[OUTPUT_DIM];

	CoarseGrainedScheduler* scheduler;

public:
	NeuralNetworkCoarseGrained(CoarseGrainedScheduler* sched, unsigned int fixed_seed = 42u) : scheduler(sched) {
		// RNG determinista (evita rand()/srand())
		std::mt19937 rng;
		rng.seed(fixed_seed);
		std::uniform_real_distribution<double> uni(0.0, 1.0);
		for (int i = 0; i < INPUT_DIM; i++) {
			for (int j = 0; j < HIDDEN_NEURONS; j++) {
				wh[i][j] = (uni(rng) - 0.5) * sqrt(2.0 / INPUT_DIM);
			}
		}
		for (int j = 0; j < HIDDEN_NEURONS; j++) bh[j] = 0.0;
		for (int i = 0; i < HIDDEN_NEURONS; i++) {
			for (int j = 0; j < OUTPUT_DIM; j++) {
				wo[i][j] = (uni(rng) - 0.5) * sqrt(2.0 / HIDDEN_NEURONS);
			}
		}
		for (int j = 0; j < OUTPUT_DIM; j++) bo[j] = 0.0;
	}

	// Forward propagation (CGMT)
	double forward_cgmt(const double input[]) {
		vector<thread> threads;
		int num_threads = scheduler->get_num_threads();

		// Capa oculta
		scheduler->begin_phase();
		auto compute_hidden = [&](int tid) {
			while (scheduler->is_phase_open()) {
				int j = scheduler->take_next(HIDDEN_NEURONS);
				if (j >= 0) {
					hidden_input[j] = bh[j];
					for (int i = 0; i < INPUT_DIM; i++) {
						hidden_input[j] += scheduler->mul_cgmt_light(tid, input[i], wh[i][j], DOT_PRODUCT, 10u, j, i);
					}
					hidden_output[j] = tanh_activation(hidden_input[j]);
				} else {
					scheduler->mark_done(tid);
					scheduler->idle_cgmt(tid);
				}
			}
		};

		for (int t = 0; t < num_threads; t++) threads.emplace_back(compute_hidden, t);
		for (auto& t : threads) t.join();

		// Capa de salida
		scheduler->begin_phase();
		for (int outj = 0; outj < OUTPUT_DIM; outj++) output_input[outj] = bo[outj];

		threads.clear();
		auto compute_output = [&](int tid) {
			while (scheduler->is_phase_open()) {
				int i = scheduler->take_next(HIDDEN_NEURONS);
				if (i >= 0) {
					for (int outj = 0; outj < OUTPUT_DIM; outj++) {
						output_input[outj] += scheduler->mul_cgmt_light(tid, hidden_output[i], wo[i][outj], DOT_PRODUCT, 20u, i, outj);
					}
				} else {
					scheduler->mark_done(tid);
					scheduler->idle_cgmt(tid);
				}
			}
		};

		for (int t = 0; t < num_threads; t++) threads.emplace_back(compute_output, t);
		for (auto& t : threads) t.join();

		for (int outj = 0; outj < OUTPUT_DIM; outj++) output[outj] = output_input[outj];
		return output[0];
	}

	// Backpropagation (CGMT)
	void backward_cgmt(const double input[], double target, double learning_rate) {
		double output_delta[OUTPUT_DIM];
		for (int j = 0; j < OUTPUT_DIM; j++) output_delta[j] = output[j] - target;

		double hidden_error[HIDDEN_NEURONS];
		double hidden_delta[HIDDEN_NEURONS];

		int num_threads = scheduler->get_num_threads();
		vector<thread> threads;

		// Cálculo de hidden_error y hidden_delta
		scheduler->begin_phase();
		auto compute_hidden_error = [&](int tid) {
			while (scheduler->is_phase_open()) {
				int j = scheduler->take_next(HIDDEN_NEURONS);
				if (j >= 0) {
					hidden_error[j] = 0.0;
					for (int k = 0; k < OUTPUT_DIM; k++) {
						hidden_error[j] += scheduler->mul_cgmt_heavy(tid, output_delta[k], wo[j][k], DOT_PRODUCT, 30u, j, k);
					}
					double deriv = tanh_derivative(hidden_input[j]);
					hidden_delta[j] = scheduler->mul_cgmt_heavy(tid, hidden_error[j], deriv, ACTIVATION, 31u, j, 0);
				} else {
					scheduler->mark_done(tid);
					scheduler->idle_cgmt(tid);
				}
			}
		};

		for (int t = 0; t < num_threads; t++) threads.emplace_back(compute_hidden_error, t);
		for (auto& t : threads) t.join();

		// Actualización de wo
		scheduler->begin_phase();
		threads.clear();
		auto update_wo = [&](int tid) {
			while (scheduler->is_phase_open()) {
				int i = scheduler->take_next(HIDDEN_NEURONS);
				if (i >= 0) {
					for (int j = 0; j < OUTPUT_DIM; j++) {
						double grad = scheduler->mul_cgmt_heavy(tid, output_delta[j], hidden_output[i], DOT_PRODUCT, 40u, i, j);
						wo[i][j] -= scheduler->mul_total(learning_rate, grad);
					}
				} else {
					scheduler->mark_done(tid);
					scheduler->idle_cgmt(tid);
				}
			}
		};
		for (int t = 0; t < num_threads; t++) threads.emplace_back(update_wo, t);
		for (auto& t : threads) t.join();

		// Bias de salida
		for (int j = 0; j < OUTPUT_DIM; j++) bo[j] -= scheduler->mul_total(learning_rate, output_delta[j]);

		// Actualización de wh
		scheduler->begin_phase();
		threads.clear();
		auto update_wh = [&](int tid) {
			while (scheduler->is_phase_open()) {
				int j = scheduler->take_next(HIDDEN_NEURONS);
				if (j >= 0) {
					for (int i = 0; i < INPUT_DIM; i++) {
						double grad = scheduler->mul_cgmt_heavy(tid, hidden_delta[j], input[i], DOT_PRODUCT, 50u, j, i);
						wh[i][j] -= scheduler->mul_total(learning_rate, grad);
					}
				} else {
					scheduler->mark_done(tid);
					scheduler->idle_cgmt(tid);
				}
			}
		};
		for (int t = 0; t < num_threads; t++) threads.emplace_back(update_wh, t);
		for (auto& t : threads) t.join();

		// Bias de capa oculta
		for (int j = 0; j < HIDDEN_NEURONS; j++) bh[j] -= scheduler->mul_total(learning_rate, hidden_delta[j]);
	}

	void train(const vector<vector<double>>& X, const vector<double>& Y, int epochs, double learning_rate) {
		int n_samples = (int)X.size();
		for (int epoch = 0; epoch < epochs; epoch++) {
			double total_loss = 0.0;
			for (int i = 0; i < n_samples; i++) {
				double in[INPUT_DIM];
				for (int j = 0; j < INPUT_DIM; j++) in[j] = X[i][j];

				double prediction = forward_cgmt(in);
				double error = prediction - Y[i];
				total_loss += scheduler->mul_total(error, error);

				backward_cgmt(in, Y[i], learning_rate);
			}

			if ((epoch + 1) % 500 == 0) {
				double mse = total_loss / n_samples;
				cout << "Epoch " << (epoch + 1) << "/" << epochs << " - MSE: " << mse << endl;
			}
		}
	}

	double predict(const double input[]) { return forward_cgmt(input); }

	double evaluate(const vector<vector<double>>& X, const vector<double>& Y) {
		double total_loss = 0.0;
		int n_samples = (int)X.size();
		for (int i = 0; i < n_samples; i++) {
			double in[INPUT_DIM];
			for (int j = 0; j < INPUT_DIM; j++) in[j] = X[i][j];
			double prediction = forward_cgmt(in);
			double error = prediction - Y[i];
			total_loss += scheduler->mul_total(error, error);
		}
		return total_loss / n_samples;
	}

	CoarseGrainedScheduler* get_scheduler() { return scheduler; }
};