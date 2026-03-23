// Algoritmo de entrenamiento Round-Robin (coarse-grained multithreading - CGMT) con hilos.
// - Solo hay 1 hilo "activo" a la vez (los demás esperan).
// - El cambio de contexto se evalúa cuando ocurre un stall.
// - En esta implementación, los stalls se disparan de forma NO determinística: cada N
//   multiplicaciones "pesadas" (N aleatorio por hilo) ocurre un stall.
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <iomanip>

#include "../../common/nn_config.h"
#include "../../common/nn_math.h"
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
	int heavy_switch_min;
	int heavy_switch_max;
	int skip_cs_chance_percent; // Probabilidad (0-100) de NO hacer context switch cuando hay stall.

	// Métricas
	long long cycle_mul_light_count;
	long long cycle_mul_heavy_count;
	long long uncounted_mul_count;
	long long compute_multiplications; // (legacy) livianas + pesadas
	long long total_multiplications;
	long long stall_count;
	long long stall_events_on_heavy_count;
	long long context_switch_count;
	long long nop_count;
	long long global_clock;

	// Estado
	int current_active_thread;
	bool phase_open;
	int phase_done_count;
	vector<bool> thread_done;
	vector<long long> active_heavy_since_switch;
	vector<unsigned int> prng_state;
	vector<int> next_heavy_switch_threshold;

	mutex m;
	condition_variable cv;

	static inline unsigned int xorshift32(unsigned int& state) {
		state ^= state << 13;
		state ^= state >> 17;
		state ^= state << 5;
		return state;
	}

	int next_threshold_for_thread_locked(int thread_id) {
		// Rango inclusivo [min, max]. Si vienen invertidos, se normaliza.
		int lo = heavy_switch_min;
		int hi = heavy_switch_max;
		if (lo < 1) lo = 1;
		if (hi < 1) hi = 1;
		if (lo > hi) std::swap(lo, hi);
		unsigned int r = xorshift32(prng_state[thread_id]);
		int span = (hi - lo + 1);
		return lo + (int)(r % (unsigned int)span);
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
						   long long context_switch_penalty_cycles_arg = 1)
		: num_threads(num_threads_arg), stall_latency_cycles(stall_latency_cycles_arg),
		  context_switch_penalty_cycles(context_switch_penalty_cycles_arg),
		  heavy_switch_min(2), heavy_switch_max(8), skip_cs_chance_percent(15),
		  cycle_mul_light_count(0), cycle_mul_heavy_count(0), uncounted_mul_count(0),
		  compute_multiplications(0), total_multiplications(0),
		  stall_count(0), stall_events_on_heavy_count(0), context_switch_count(0), nop_count(0),
		  global_clock(0), current_active_thread(0), phase_open(false),
		  phase_done_count(0), thread_done(num_threads_arg, false),
		  active_heavy_since_switch(num_threads_arg, 0),
		  prng_state(num_threads_arg, 0u),
		  next_heavy_switch_threshold(num_threads_arg, 0) {
		// Semillas por hilo: mezcla de tiempo + id del hilo lógico (determinístico por corrida).
		unsigned int base = (unsigned int)time(NULL);
		for (int i = 0; i < num_threads; i++) {
			prng_state[i] = base ^ (0x9E3779B9u * (unsigned int)(i + 1));
			next_heavy_switch_threshold[i] = 2; // Se recalcula en begin_phase().
		}
	}

	// Configura qué tan seguido se OMITE el context switch cuando hay un stall.
	// Ej.: percent=15 => ~15% de stalls NO harán switch.
	void set_skip_context_switch_chance_percent(int percent) {
		lock_guard<mutex> lock(m);
		if (percent < 0) percent = 0;
		if (percent > 100) percent = 100;
		skip_cs_chance_percent = percent;
	}

	// Permite configurar el rango aleatorio de N (cada cuántas multiplicaciones pesadas se produce un stall).
	void set_heavy_switch_random_range(int min_n, int max_n) {
		lock_guard<mutex> lock(m);
		heavy_switch_min = min_n;
		heavy_switch_max = max_n;
		// thresholds se recalculan al begin_phase
	}

	void begin_phase() {
		unique_lock<mutex> lock(m);
		phase_open = true;
		phase_done_count = 0;
		current_active_thread = 0;
		fill(thread_done.begin(), thread_done.end(), false);
		fill(active_heavy_since_switch.begin(), active_heavy_since_switch.end(), 0);
		for (int i = 0; i < num_threads; i++) {
			next_heavy_switch_threshold[i] = next_threshold_for_thread_locked(i);
		}
		cv.notify_all();
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
	T mul_cgmt_impl(int thread_id, T a, T b, bool is_heavy) {
		while (true) {
			wait_active(thread_id);
			if (!is_phase_open()) return T{};

			unique_lock<mutex> lock(m);
			if (!phase_open) return T{};
			if (current_active_thread != thread_id) continue;

			// Ejecutar 1 multiplicación del hilo activo (cuenta como 1 ciclo).
			compute_multiplications++;
			total_multiplications++;
			global_clock++;
			T r = a * b;

			if (is_heavy) {
				cycle_mul_heavy_count++;
				active_heavy_since_switch[thread_id]++;

				// Stall NO determinístico: cada N multiplicaciones pesadas (N aleatorio) ocurre un stall.
				// En cada stall, casi siempre hay context switch, pero a veces (aleatorio) NO se hace.
				if (active_heavy_since_switch[thread_id] >= (long long)next_heavy_switch_threshold[thread_id]) {
					// Ocurre stall.
					stall_count++;
					stall_events_on_heavy_count++;
					active_heavy_since_switch[thread_id] = 0;
					next_heavy_switch_threshold[thread_id] = next_threshold_for_thread_locked(thread_id);

					// Costo del stall (ciclos sin computar).
					if (stall_latency_cycles > 0) {
						global_clock += stall_latency_cycles;
					}

					// En un stall, normalmente hacemos context switch, pero con probabilidad
					// skip_cs_chance_percent lo omitimos.
					unsigned int rr = xorshift32(prng_state[thread_id]);
					bool skip_cs = ((int)(rr % 100u)) < skip_cs_chance_percent;

					if (!skip_cs) {
						int old = current_active_thread;
						if (unfinished_threads_locked() > 1) {
							switch_to_next_thread_locked();
							if (current_active_thread != old) {
								context_switch_count++;
								if (context_switch_penalty_cycles > 0) {
									global_clock += context_switch_penalty_cycles;
								}
							}
						}
					}
					cv.notify_all();
				}
			} else {
				cycle_mul_light_count++;
			}

			return r;
		}
	}

	// 2) Multiplicación liviana: se cuenta como ciclo (NO cambia de hilo).
	template <typename T>
	T mul_cgmt_light(int thread_id, T a, T b) {
		return mul_cgmt_impl(thread_id, a, b, false);
	}

	// 3) Multiplicación pesada: se cuenta como ciclo y puede provocar stalls (y, por ende, switches).
	template <typename T>
	T mul_cgmt_heavy(int thread_id, T a, T b) {
		return mul_cgmt_impl(thread_id, a, b, true);
	}

	template <typename T>
	T mul_cgmt(int thread_id, T a, T b) {
		// Compatibilidad: por defecto se considera liviana.
		return mul_cgmt_light(thread_id, a, b);
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
		cout << "multiplicaciones no contadas (registradas): " << uncounted_mul_count << endl;
		cout << "multiplicaciones totales (registradas): " << total_multiplications << endl;
		cout << "stalls (por heavy cada N aleatorio): " << stall_count << endl;
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
	NeuralNetworkCoarseGrained(CoarseGrainedScheduler* sched) : scheduler(sched) {
		srand(time(NULL));
		for (int i = 0; i < INPUT_DIM; i++) {
			for (int j = 0; j < HIDDEN_NEURONS; j++) {
				wh[i][j] = ((double)rand() / RAND_MAX - 0.5) * sqrt(2.0 / INPUT_DIM);
			}
		}
		for (int j = 0; j < HIDDEN_NEURONS; j++) bh[j] = 0.0;
		for (int i = 0; i < HIDDEN_NEURONS; i++) {
			for (int j = 0; j < OUTPUT_DIM; j++) {
				wo[i][j] = ((double)rand() / RAND_MAX - 0.5) * sqrt(2.0 / HIDDEN_NEURONS);
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
			int j = tid;
			int step = num_threads;
			while (scheduler->is_phase_open()) {
				if (j < HIDDEN_NEURONS) {
					hidden_input[j] = bh[j];
					for (int i = 0; i < INPUT_DIM; i++) {
						hidden_input[j] += scheduler->mul_cgmt_light(tid, input[i], wh[i][j]);
					}
					hidden_output[j] = tanh_activation(hidden_input[j]);
					j += step;
					if (j >= HIDDEN_NEURONS) scheduler->mark_done(tid);
				} else {
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
			int i = tid;
			int step = num_threads;
			while (scheduler->is_phase_open()) {
				if (i < HIDDEN_NEURONS) {
					for (int outj = 0; outj < OUTPUT_DIM; outj++) {
						output_input[outj] += scheduler->mul_cgmt_light(tid, hidden_output[i], wo[i][outj]);
					}
					i += step;
					if (i >= HIDDEN_NEURONS) scheduler->mark_done(tid);
				} else {
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
			int j = tid;
			int step = num_threads;
			while (scheduler->is_phase_open()) {
				if (j < HIDDEN_NEURONS) {
					hidden_error[j] = 0.0;
					for (int k = 0; k < OUTPUT_DIM; k++) {
						hidden_error[j] += scheduler->mul_cgmt_heavy(tid, output_delta[k], wo[j][k]);
					}
					double deriv = tanh_derivative(hidden_input[j]);
					hidden_delta[j] = scheduler->mul_cgmt_heavy(tid, hidden_error[j], deriv);
					j += step;
					if (j >= HIDDEN_NEURONS) scheduler->mark_done(tid);
				} else {
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
			int i = tid;
			int step = num_threads;
			while (scheduler->is_phase_open()) {
				if (i < HIDDEN_NEURONS) {
					for (int j = 0; j < OUTPUT_DIM; j++) {
						double grad = scheduler->mul_cgmt_heavy(tid, output_delta[j], hidden_output[i]);
						wo[i][j] -= scheduler->mul_total(learning_rate, grad);
					}
					i += step;
					if (i >= HIDDEN_NEURONS) scheduler->mark_done(tid);
				} else {
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
			int j = tid;
			int step = num_threads;
			while (scheduler->is_phase_open()) {
				if (j < HIDDEN_NEURONS) {
					for (int i = 0; i < INPUT_DIM; i++) {
						double grad = scheduler->mul_cgmt_heavy(tid, hidden_delta[j], input[i]);
						wh[i][j] -= scheduler->mul_total(learning_rate, grad);
					}
					j += step;
					if (j >= HIDDEN_NEURONS) scheduler->mark_done(tid);
				} else {
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

