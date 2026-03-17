//algoritmo de entrenamiento roundrobin -fine grained con hilos 
//el context switch se realiza para cada multiplicación

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

// Constantes
const int INPUT_DIM = 3;
const int HIDDEN_NEURONS = 30;
const int OUTPUT_DIM = 1;
const double LEARNING_RATE = 0.08;
const int EPOCHS = 1500;

// activación
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

//scheduler por round robin
class RoundRobinScheduler {
private:
    int num_threads;
    long long quantum;
    long long multiplication_count;
    long long total_multiplications;
    long long nop_count;
    long long global_clock;
    int current_turn;
    mutex m;
    condition_variable cv;
    int phase_done_count;
    bool phase_open;
    
public:
    RoundRobinScheduler(int num_threads_arg, long long quantum_size = 1) 
    : num_threads(num_threads_arg), quantum(quantum_size),
    multiplication_count(0), total_multiplications(0), nop_count(0),
    global_clock(0), current_turn(0),
    phase_done_count(0), phase_open(false) {}

    void begin_phase() {
        unique_lock<mutex> lock(m);
        phase_open = true;
        phase_done_count = 0;
        current_turn = 0;
        cv.notify_all();
    }

    //para cuando termina un thread
    void mark_done() {
        unique_lock<mutex> lock(m);
        if (!phase_open) return;
        phase_done_count++;
        if (phase_done_count >= num_threads) {
            phase_open = false;
        }
        cv.notify_all();
    }

    bool is_phase_open() {
        lock_guard<mutex> lock(m);
        return phase_open;
    }

    //bloquea el turno hasta que sea el turno del thread_id
    void wait_turn(int thread_id) {
        unique_lock<mutex> lock(m);
        cv.wait(lock, [&] {
            if (!phase_open) return true;
            return current_turn == thread_id;
        });
    }

    //Avance turno
    void advance_turn() {
        unique_lock<mutex> lock(m);
        global_clock++;
    current_turn = (current_turn + 1) % num_threads;
        cv.notify_all();
    }

    //operacion que toma 1 "ciclo", espera turno, ejecuta 1 multiplicacion, y rota(context switch)
    template <typename T>
    T mul_fgmt(int thread_id, T a, T b) {
        wait_turn(thread_id);
        // Si la fase cerró mientras esperábamos, devolvemos algo neutral.
        if (!is_phase_open()) return T{};
        multiplication_count++;
        total_multiplications++;
        T r = a * b;
        advance_turn();
        return r;
    }

    // Multiplicación "normal": se cuenta, pero NO fuerza turnos ni consume clock.
    template <typename T>
    T mul_total(T a, T b) {
        total_multiplications++;
        return a * b;
    }

    //NOP: consume 1 ciclo y rota turno aunque este thread no tenga trabajo
    void idle_fgmt(int thread_id) {
        wait_turn(thread_id);
        if (!is_phase_open()) return;
    nop_count++;
        advance_turn();
    }
    
    long long get_multiplication_count() const {
        return multiplication_count;
    }

    long long get_total_multiplications() const { return total_multiplications; }

    long long get_nop_count() const { return nop_count; }

    long long get_global_clock() const { return global_clock; }
    
    int get_num_threads() const { return num_threads; }
    
    void print_stats() {
        cout << "\n --- metricas ---" << endl;
        cout << "multiplicaciones/ciclos: " << multiplication_count << endl;
        cout << "multiplicaciones totales: " << total_multiplications << endl;
        cout << "NOPs: " << nop_count << endl;
        cout << "ciclos simulados: " << global_clock << endl;
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
    
public:
    NeuralNetworkFineGrained(RoundRobinScheduler* sched) : scheduler(sched) {
        srand(time(NULL));
        
        // Inicializar pesos input -> hidden
        for (int i = 0; i < INPUT_DIM; i++) {
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                wh[i][j] = ((double)rand() / RAND_MAX - 0.5) * sqrt(2.0 / INPUT_DIM);
            }
        }
        
        // Inicializar bias hidden
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            bh[j] = 0.0;
        }
        
        // Inicializar pesos hidden -> output
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            for (int j = 0; j < OUTPUT_DIM; j++) {
                wo[i][j] = ((double)rand() / RAND_MAX - 0.5) * sqrt(2.0 / HIDDEN_NEURONS);
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

    /* este scheduler va por neuronas y dentro de cada neurona se realiza el forward que dentro del forward
    está el scheduler que hace el roundrobin para las multiplicicaciones
    */
    scheduler->begin_phase();
        auto compute_hidden = [&](int thread_id) {
            int start = thread_id;
            int step = num_threads;

            int j = start;
            while (scheduler->is_phase_open()) {
                if (j < HIDDEN_NEURONS) {
                    // COMPUTE una neurona
                    hidden_input[j] = bh[j];
                    for (int i = 0; i < INPUT_DIM; i++) {
                        hidden_input[j] += scheduler->mul_fgmt(thread_id, input[i], wh[i][j]);
                    }
                    hidden_output[j] = tanh_activation(hidden_input[j]);
                    j += step;

                    if (j >= HIDDEN_NEURONS) {
                        scheduler->mark_done();
                    }
                } else {
                    scheduler->idle_fgmt(thread_id);
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
            int start_idx = thread_id;
            int step_idx = num_threads;
            int i = start_idx;

            while (scheduler->is_phase_open()) {
                if (i < HIDDEN_NEURONS) {
                    for (int outj = 0; outj < OUTPUT_DIM; outj++) {
                        output_input[outj] += scheduler->mul_fgmt(thread_id, hidden_output[i], wo[i][outj]);
                    }
                    i += step_idx;
                    if (i >= HIDDEN_NEURONS) {
                        scheduler->mark_done();
                    }
                } else {
                    scheduler->idle_fgmt(thread_id);
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

        // hidden_error
        //se hace con el scheduler pq como analiza para atras ocupa aún repartir el trabajo por neurona 
        int num_threads = scheduler->get_num_threads();
        scheduler->begin_phase();
        vector<thread> threads;
        auto compute_hidden_error = [&](int thread_id) {
            int j = thread_id;
            int step = num_threads;
            while (scheduler->is_phase_open()) {
                if (j < HIDDEN_NEURONS) {
                    hidden_error[j] = 0.0;
                    for (int k = 0; k < OUTPUT_DIM; k++) {
                        hidden_error[j] += scheduler->mul_fgmt(thread_id, output_delta[k], wo[j][k]);
                    }

                    double t = tanh(hidden_input[j]);
                    // Estas multiplicaciones forman parte del cómputo por neurona (delta),
                    // así que también pasan por el round-robin.
                    double deriv = 1.0 - scheduler->mul_fgmt(thread_id, t, t);

                    hidden_delta[j] = scheduler->mul_fgmt(thread_id, hidden_error[j], deriv);
                    j += step;
                    if (j >= HIDDEN_NEURONS) scheduler->mark_done();
                } else {
                    scheduler->idle_fgmt(thread_id);
                }
            }
        };

        for (int t = 0; t < num_threads; t++) threads.emplace_back(compute_hidden_error, t);
        for (auto& t : threads) t.join();
        
        //pesos de hidden - output
        scheduler->begin_phase();
        threads.clear();
        auto update_wo = [&](int thread_id) {
            int i = thread_id;
            int step = num_threads;
            while (scheduler->is_phase_open()) {
                if (i < HIDDEN_NEURONS) {
                    for (int j = 0; j < OUTPUT_DIM; j++) {
                        // grad = output_delta * hidden_output (FGMT)
                        double grad = scheduler->mul_fgmt(thread_id, output_delta[j], hidden_output[i]);
                        // learning_rate * grad (total)
                        wo[i][j] -= scheduler->mul_total(learning_rate, grad);
                    }
                    i += step;
                    if (i >= HIDDEN_NEURONS) scheduler->mark_done();
                } else {
                    scheduler->idle_fgmt(thread_id);
                }
            }
        };
        for (int t = 0; t < num_threads; t++) threads.emplace_back(update_wo, t);
        for (auto& t : threads) t.join();
        
        //bias output
        for (int j = 0; j < OUTPUT_DIM; j++) {
            // lr * delta (total)
            bo[j] -= scheduler->mul_total(learning_rate, output_delta[j]);
        }
        
        //pesos input - hidden
        scheduler->begin_phase();
        threads.clear();
        auto update_wh = [&](int thread_id) {
            int j = thread_id;
            int step = num_threads;
            while (scheduler->is_phase_open()) {
                if (j < HIDDEN_NEURONS) {
                    for (int i = 0; i < INPUT_DIM; i++) {
                        double grad = scheduler->mul_fgmt(thread_id, hidden_delta[j], input[i]);
                        wh[i][j] -= scheduler->mul_total(learning_rate, grad);
                    }
                    j += step;
                    if (j >= HIDDEN_NEURONS) scheduler->mark_done();
                } else {
                    scheduler->idle_fgmt(thread_id);
                }
            }
        };
        for (int t = 0; t < num_threads; t++) threads.emplace_back(update_wh, t);
        for (auto& t : threads) t.join();
        
        // bias hidden
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