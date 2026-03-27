// Algoritmo de entrenamiento con scheduler coarse-grained (CGMT).
//
// Diferencia clave respecto a FGMT:
//   FGMT → context switch después de CADA multiplicación (round-robin estricto).
//   CGMT → el hilo activo ejecuta indefinidamente; solo cede el turno cuando
//           ocurre un stall por cache miss.
//
// Ventaja de CGMT: cuando no hay stall, no hay overhead de context switch.
// El stall hiding funciona igual que en FGMT: si hay otro hilo disponible,
// ese hilo cubre la latencia del stall mientras el primero espera.
#include <iostream>
#include <cstdlib>
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

// CoarseGrainedScheduler: modela un procesador donde un solo hilo ejecuta
// a la vez y solo lo cede cuando sufre un cache miss (stall).
// Con múltiples hilos, el stall puede quedar oculto si el otro hilo hace
// suficiente trabajo antes de devolver el turno → hiding de latencia.
class CoarseGrainedScheduler {
private:
	int num_threads;
	long long stall_latency_cycles;
	long long context_switch_penalty_cycles;

	// Latencias en ciclos por tipo de operación (mismos valores que FGMT
	// para que las comparaciones entre variantes sean válidas).
	//   LIGHT_DOT = 1: forward pass pipelined → 1 ciclo.
	//   HEAVY_DOT = 3: backward pass, 3 etapas de pipeline.
	//   HEAVY_ACT = 2: derivada de tanh (t*t, error*(1-tt)), deps FP → 2 ciclos.
	static const long long LATENCY_LIGHT_DOT = 1;
	static const long long LATENCY_HEAVY_DOT = 3;
	static const long long LATENCY_HEAVY_ACT = 2;

	long long cycle_mul_light_count;
	long long cycle_mul_heavy_count;
	long long stall_count;
	long long context_switch_count;
	long long global_clock;

	uint32_t stall_seed;

	int current_active_thread;
	bool phase_open;
	int phase_done_count;
	vector<bool> thread_done;
	vector<long long> thread_resume_clock;  // ciclo en que cada hilo puede reanudar tras un stall

	mutex m;
	condition_variable cv;

	// Hash determinista para simular cache misses reproducibles.
	// Misma función que FGMT: dado el mismo (seed, tag, j, k, op),
	// siempre produce el mismo resultado → stalls idénticos por evento.
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

	// Latencia de context switch aplicada cuando hay un stall y se cambia de hilo.
	// En CGMT esto solo ocurre en el momento del stall, no en cada multiplicación.
	long long pick_latency(bool is_heavy, OpType op) const {
		if (!is_heavy) return LATENCY_LIGHT_DOT;
		return (op == ACTIVATION) ? LATENCY_HEAVY_ACT : LATENCY_HEAVY_DOT;
	}

	// Avanza current_active_thread al siguiente hilo que aún no terminó.
	// Se llama solo cuando el hilo activo sufre un stall o llama a mark_done().
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

	// Cuántos hilos no han llamado mark_done() todavía en esta fase.
	// Se usa para decidir si hay otro hilo disponible para hacer hiding.
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
	                       uint32_t stall_seed_arg = 42u)
		: num_threads(num_threads_arg),
		  stall_latency_cycles(stall_latency_cycles_arg),
		  context_switch_penalty_cycles(context_switch_penalty_cycles_arg),
		  stall_seed(stall_seed_arg),
		  cycle_mul_light_count(0), cycle_mul_heavy_count(0),
		  stall_count(0), context_switch_count(0),
		  global_clock(0), current_active_thread(0), phase_open(false),
		  phase_done_count(0), thread_done(num_threads_arg, false),
		  thread_resume_clock(num_threads_arg, 0LL) {}

	// Abre una nueva fase: todos los hilos compiten por ser el activo inicial (hilo 0).
	void begin_phase() {
		unique_lock<mutex> lock(m);
		phase_open = true;
		phase_done_count = 0;
		current_active_thread = 0;
		fill(thread_done.begin(), thread_done.end(), false);
		fill(thread_resume_clock.begin(), thread_resume_clock.end(), 0LL);
		cv.notify_all();
	}

	// El hilo llama a mark_done() cuando termina todas sus neuronas.
	// Si era el activo, el scheduler pasa el turno al siguiente hilo disponible.
	// Cuando todos terminan, la fase se cierra automáticamente.
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

	long long get_light_count()          const { return cycle_mul_light_count; }
	long long get_heavy_count()          const { return cycle_mul_heavy_count; }
	long long get_stall_count()          const { return stall_count; }
	long long get_context_switch_count() const { return context_switch_count; }
	long long get_global_clock()         const { return global_clock; }

	// Bloqueo CGMT: el hilo espera hasta ser el activo.
	// Diferencia con FGMT: aquí el hilo puede ejecutar muchas multiplicaciones
	// seguidas sin ceder; solo espera si otro hilo tiene el turno.
	// Si el hilo tuvo un stall y el otro no cubrió toda la latencia,
	// se pagan los ciclos restantes al retomar el control.
	void wait_active(int thread_id) {
		unique_lock<mutex> lock(m);
		cv.wait(lock, [&] { return !phase_open || current_active_thread == thread_id; });
		if (phase_open && global_clock < thread_resume_clock[thread_id]) {
			global_clock = thread_resume_clock[thread_id];
		}
	}

	// Núcleo de la multiplicación CGMT (compartido por light y heavy).
	// El hilo activo ejecuta la multiplicación y luego evalúa si hubo stall.
	// Si hay stall y otro hilo disponible:
	//   → se registra thread_resume_clock para el hiding,
	//   → se cambia al otro hilo (context switch + su latencia),
	//   → el otro hilo "cubre" los ciclos del stall mientras ejecuta.
	// Si no hay otro hilo:
	//   → se paga el stall completo sin hiding.
	template <typename T>
	T mul_cgmt_impl(int thread_id, T a, T b, bool is_heavy, OpType op, uint32_t tag, int j, int k) {
		while (true) {
			wait_active(thread_id);
			if (!is_phase_open()) return T{};

			unique_lock<mutex> lock(m);
			if (!phase_open) return T{};
			if (current_active_thread != thread_id) continue;

			global_clock++;
			T r = a * b;

			if (is_heavy) cycle_mul_heavy_count++;
			else          cycle_mul_light_count++;

			// Probabilidades de cache miss (mismas que FGMT para comparabilidad):
			//   DOT_PRODUCT: 2.4% de miss → 8 ciclos de stall
			//   ACTIVATION:  0.6% de miss → 3 ciclos de stall
			uint32_t miss_pct1000 = (op == DOT_PRODUCT) ? 24u : 6u;
			long long stall_lat   = (op == DOT_PRODUCT) ? 8LL : 3LL;
			uint32_t pct = random(stall_seed, tag, j, k, op) % 1000u;
			if (pct < miss_pct1000) {
				stall_count++;
				int old = current_active_thread;
				if (unfinished_threads_locked() > 1) {
					thread_resume_clock[thread_id] = global_clock + stall_lat;
					switch_to_next_thread_locked();
					if (current_active_thread != old) {
						context_switch_count++;
						global_clock += pick_latency(is_heavy, op);
					}
				} else {
					global_clock += stall_lat;
				}
				cv.notify_all();
			}

			return r;
		}
	}

	// mul_cgmt_light: multiplicación forward pass (w*x).
	// No cambia de hilo salvo cache miss; contribuye a los ciclos del reloj.
	template <typename T>
	T mul_cgmt_light(int thread_id, T a, T b, OpType op, uint32_t tag, int j, int k) {
		return mul_cgmt_impl(thread_id, a, b, false, op, tag, j, k);
	}

	// mul_cgmt_heavy: multiplicación backward pass (w*δ, t*t, error*(1-tt)).
	// Misma lógica que light pero con mayor latencia de context switch y
	// mayor probabilidad de cache miss (gradientes, peor localidad).
	template <typename T>
	T mul_cgmt_heavy(int thread_id, T a, T b, OpType op, uint32_t tag, int j, int k) {
		return mul_cgmt_impl(thread_id, a, b, true, op, tag, j, k);
	}

	// mul_total: operaciones fuera del modelo de ciclos (lr*grad, error²).
	// No consume global_clock ni provoca stalls; es solo una multiplicación escalar.
	template <typename T>
	T mul_total(T a, T b) {
		return a * b;
	}
};

// Red neuronal de una capa oculta con scheduler CGMT.
// Estructura idéntica a la variante FGMT, con dos diferencias:
//   1. Se usa CoarseGrainedScheduler en lugar de RoundRobinScheduler.
//   2. Cada lambda llama a mark_done() al terminar, en lugar de
//      yield_turns_until_closed() + atómico, porque en CGMT el hilo
//      activo simplemente se cede cuando marca que terminó.
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
	// Inicialización de Xavier con seed fija para resultados reproducibles
	// y comparables entre variantes (normal, cgmt, fgmt con misma seed).
	NeuralNetworkCoarseGrained(CoarseGrainedScheduler* sched, unsigned int fixed_seed = 42u) : scheduler(sched) {
		std::mt19937 rng;
		rng.seed(fixed_seed);
		std::uniform_real_distribution<double> uni(0.0, 1.0);
		for (int i = 0; i < INPUT_DIM; i++)
			for (int j = 0; j < HIDDEN_NEURONS; j++)
				wh[i][j] = (uni(rng) - 0.5) * sqrt(2.0 / INPUT_DIM);
		for (int j = 0; j < HIDDEN_NEURONS; j++) bh[j] = 0.0;
		for (int i = 0; i < HIDDEN_NEURONS; i++)
			for (int j = 0; j < OUTPUT_DIM; j++)
				wo[i][j] = (uni(rng) - 0.5) * sqrt(2.0 / HIDDEN_NEURONS);
		for (int j = 0; j < OUTPUT_DIM; j++) bo[j] = 0.0;
	}

	// Forward pass en dos fases separadas por un join().
	// Fase 1 (hidden): cada hilo calcula sus neuronas ocultas (w*x + b → tanh).
	// Fase 2 (output): cada hilo acumula su parte de la suma ponderada de hidden.
	// El join actúa como barrera: fase 2 no empieza hasta que todos los hilos
	// de fase 1 llamaron a mark_done().
	double forward_cgmt(const double input[]) {
		vector<thread> threads;
		int num_threads = scheduler->get_num_threads();

		// --- Fase 1: capa oculta ---
		scheduler->begin_phase();
		auto compute_hidden = [&](int tid) {
			for (int j = tid; j < HIDDEN_NEURONS; j += num_threads) {
				hidden_input[j] = bh[j];
				for (int i = 0; i < INPUT_DIM; i++) {
					hidden_input[j] += scheduler->mul_cgmt_light(tid, input[i], wh[i][j], DOT_PRODUCT, 10u, j, i);
				}
				hidden_output[j] = tanh_activation(hidden_input[j]);
			}
			scheduler->mark_done(tid);
		};

		for (int t = 0; t < num_threads; t++) threads.emplace_back(compute_hidden, t);
		for (auto& t : threads) t.join();

		// --- Fase 2: capa de salida ---
		scheduler->begin_phase();
		for (int outj = 0; outj < OUTPUT_DIM; outj++) output_input[outj] = bo[outj];

		threads.clear();
		auto compute_output = [&](int tid) {
			for (int i = tid; i < HIDDEN_NEURONS; i += num_threads) {
				for (int outj = 0; outj < OUTPUT_DIM; outj++) {
					output_input[outj] += scheduler->mul_cgmt_light(tid, hidden_output[i], wo[i][outj], DOT_PRODUCT, 20u, i, outj);
				}
			}
			scheduler->mark_done(tid);
		};

		for (int t = 0; t < num_threads; t++) threads.emplace_back(compute_output, t);
		for (auto& t : threads) t.join();

		for (int outj = 0; outj < OUTPUT_DIM; outj++) output[outj] = output_input[outj];
		return output[0];
	}

	// Backward pass en cuatro fases (misma estructura que FGMT).
	//
	// Fase 1: hidden_error = woᵀ * output_delta; hidden_delta = error * tanh'(x)
	//         tanh'(x) = 1 - tanh²(x) modelado como 2 mul_heavy de ACTIVATION.
	// Fase 2: wo -= lr * output_delta * hidden_outputᵀ
	// Fase 3: wh -= lr * hidden_delta * inputᵀ
	// Fase 4: bias (mul_total, fuera de ciclos)
	void backward_cgmt(const double input[], double target, double learning_rate) {
		// Gradiente de MSE: dL/dy = y - target
		double output_delta[OUTPUT_DIM];
		for (int j = 0; j < OUTPUT_DIM; j++) output_delta[j] = output[j] - target;

		double hidden_error[HIDDEN_NEURONS];
		double hidden_delta[HIDDEN_NEURONS];

		int num_threads = scheduler->get_num_threads();
		vector<thread> threads;

		// --- Fase 1: error y delta de la capa oculta ---
		scheduler->begin_phase();
		auto compute_hidden_error = [&](int tid) {
			for (int j = tid; j < HIDDEN_NEURONS; j += num_threads) {
				hidden_error[j] = 0.0;
				for (int k = 0; k < OUTPUT_DIM; k++) {
					hidden_error[j] += scheduler->mul_cgmt_heavy(tid, output_delta[k], wo[j][k], DOT_PRODUCT, 30u, j, k);
				}
				// tanh'(x) = 1 - tanh²(x): dos mul_heavy de ACTIVATION porque
				// t*t y error*(1-tt) son ops FP dependientes con mayor latencia.
				double t = tanh(hidden_input[j]);
				double tt = scheduler->mul_cgmt_heavy(tid, t, t, ACTIVATION, 31u, j, 0);
				hidden_delta[j] = scheduler->mul_cgmt_heavy(tid, hidden_error[j], (1.0 - tt), ACTIVATION, 32u, j, 0);
			}
			scheduler->mark_done(tid);
		};

		for (int t = 0; t < num_threads; t++) threads.emplace_back(compute_hidden_error, t);
		for (auto& t : threads) t.join();

		// --- Fase 2: actualizar pesos hidden → output ---
		scheduler->begin_phase();
		threads.clear();
		auto update_wo = [&](int tid) {
			for (int i = tid; i < HIDDEN_NEURONS; i += num_threads) {
				for (int j = 0; j < OUTPUT_DIM; j++) {
					double grad = scheduler->mul_cgmt_heavy(tid, output_delta[j], hidden_output[i], DOT_PRODUCT, 40u, i, j);
					wo[i][j] -= scheduler->mul_total(learning_rate, grad);
				}
			}
			scheduler->mark_done(tid);
		};
		for (int t = 0; t < num_threads; t++) threads.emplace_back(update_wo, t);
		for (auto& t : threads) t.join();

		// Bias de salida fuera de ciclos
		for (int j = 0; j < OUTPUT_DIM; j++) bo[j] -= scheduler->mul_total(learning_rate, output_delta[j]);

		// --- Fase 3: actualizar pesos input → hidden ---
		scheduler->begin_phase();
		threads.clear();
		auto update_wh = [&](int tid) {
			for (int j = tid; j < HIDDEN_NEURONS; j += num_threads) {
				for (int i = 0; i < INPUT_DIM; i++) {
					double grad = scheduler->mul_cgmt_heavy(tid, hidden_delta[j], input[i], DOT_PRODUCT, 50u, j, i);
					wh[i][j] -= scheduler->mul_total(learning_rate, grad);
				}
			}
			scheduler->mark_done(tid);
		};
		for (int t = 0; t < num_threads; t++) threads.emplace_back(update_wh, t);
		for (auto& t : threads) t.join();

		// Bias de capa oculta fuera de ciclos
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
