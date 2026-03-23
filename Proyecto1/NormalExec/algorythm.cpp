//algoritmo de entrenamiento basico sin hilos

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>

using namespace std;

// Constantes de la red neuronal

const int INPUT_DIM = 3;
const int HIDDEN_NEURONS = 30;
const int OUTPUT_DIM = 1;
const double LEARNING_RATE = 0.08;
const int EPOCHS = 1500;

//se crea la funcion basefunction que espera un arreglo de datos y su dimension
double basefunction(double x[], int d){
    double sum = 0;
    for(int i=0;i<d;i++){
        sum += sin(x[i]) + 0.3*x[i]*x[i];
    }
    return sum;
}

// Función de activación tanh
double tanh_activation(double x) {
    return tanh(x);
}

// Derivada de tanh
double tanh_derivative(double x) {
    double t = tanh(x);
    return 1.0 - t * t;
}


class NeuralNetwork {
private:
    double wh[INPUT_DIM][HIDDEN_NEURONS]; // pesos de la capa de entrada a la capa oculta
    double bh[HIDDEN_NEURONS]; // bias de la capa oculta
    double wo[HIDDEN_NEURONS][OUTPUT_DIM]; // pesos de la capa oculta a la capa de salida
    double bo[OUTPUT_DIM]; // bias de la capa de salida

    // Valores intermedios para backpropagation
    double hidden_input[HIDDEN_NEURONS];   // Entrada a capa oculta (antes de activación)
    double hidden_output[HIDDEN_NEURONS];  // Salida de capa oculta (después de activación)
    double output_input[OUTPUT_DIM];       // Entrada a capa de salida
    double output[OUTPUT_DIM];             // Salida final
    
public:
    NeuralNetwork() {
        srand(time(NULL));

        //Inicia los pesos al inicio no completamente random se usa una formula para que tenga sentido conforme a la cantidad de entradas y salidas
        for (int i = 0; i < INPUT_DIM; i++) {
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                wh[i][j] = ((double)rand() / RAND_MAX - 0.5) * sqrt(2.0 / INPUT_DIM);
            }
        }
        
        // Inicializar bias hidden
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            bh[j] = 0.0;
        }
        
        // Inicializar pesos de hidden a  output
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            for (int j = 0; j < OUTPUT_DIM; j++) {
                wo[i][j] = ((double)rand() / RAND_MAX - 0.5) * sqrt(2.0 / HIDDEN_NEURONS);
            }
        }
        
        // Inicializa bias output
        for (int j = 0; j < OUTPUT_DIM; j++) {
            bo[j] = 0.0;
        }
    }

    //FORWARD PROPAGATION ---------------------------------------
    double forward(double input[]) {
        // Capa oculta
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            hidden_input[j] = bh[j];
            for (int i = 0; i < INPUT_DIM; i++) {
                hidden_input[j] += input[i] * wh[i][j];
            }
            hidden_output[j] = tanh_activation(hidden_input[j]);
        }
        
        // Capa de salida
        for (int j = 0; j < OUTPUT_DIM; j++) {
            output_input[j] = bo[j];
            for (int i = 0; i < HIDDEN_NEURONS; i++) {
                output_input[j] += hidden_output[i] * wo[i][j];
            }
            output[j] = output_input[j];
        }
        
        return output[0];
    }
    
    //BACKPROPAGATION ----------------------------------------
    void backward(double input[], double target, double learning_rate) {
        
        //error de salida (derivada de MSE)
        double output_error[OUTPUT_DIM];
        double output_delta[OUTPUT_DIM];
        
        //error medio 
        for (int j = 0; j < OUTPUT_DIM; j++) {
            output_error[j] = output[j] - target;
            output_delta[j] = output_error[j];
        }
        
        //error de capa oculta
        double hidden_error[HIDDEN_NEURONS];
        double hidden_delta[HIDDEN_NEURONS];
        
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            hidden_error[j] = 0.0;
            for (int k = 0; k < OUTPUT_DIM; k++) {
                hidden_error[j] += output_delta[k] * wo[j][k];
            }
            hidden_delta[j] = hidden_error[j] * tanh_derivative(hidden_input[j]);
        }
        
        //recalculo pesos hidden - output
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            for (int j = 0; j < OUTPUT_DIM; j++) {
                wo[i][j] -= learning_rate * output_delta[j] * hidden_output[i];
            }
        }
        
        //recalculo bias output
        for (int j = 0; j < OUTPUT_DIM; j++) {
            bo[j] -= learning_rate * output_delta[j];
        }
        
        //recalculo pesos input - hidden
        for (int i = 0; i < INPUT_DIM; i++) {
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                wh[i][j] -= learning_rate * hidden_delta[j] * input[i];
            }
        }
        
        //recalculo bias hidden
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            bh[j] -= learning_rate * hidden_delta[j];
        }
    }
    
    // ENTRENAMIENTO
    void train(vector<vector<double>>& X, vector<double>& Y, int epochs, double learning_rate) {
        int n_samples = X.size();
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double total_loss = 0.0;
            
            for (int i = 0; i < n_samples; i++) {
                double input[INPUT_DIM];
                for (int j = 0; j < INPUT_DIM; j++) {
                    input[j] = X[i][j];
                }
                
                // Forward pass
                double prediction = forward(input);
                
                // Calcular pérdida (MSE)
                double error = prediction - Y[i];
                total_loss += error * error;
                
                // Backward pass
                backward(input, Y[i], learning_rate);
            }
            
            // Mostrar progreso cada 100 épocas
            if ((epoch + 1) % 100 == 0) {
                double mse = total_loss / n_samples;
                cout << "Epoch " << (epoch + 1) << "/" << epochs 
                     << " - MSE: " << mse << endl;
            }
        }
    }

    //hace predicción
    double predict(double input[]) {
        return forward(input);
    }
    
    //evalúa que tan efectivo es (MSE)
    double evaluate(vector<vector<double>>& X, vector<double>& Y) {
        double total_loss = 0.0;
        int n_samples = X.size();
        
        for (int i = 0; i < n_samples; i++) {
            double input[INPUT_DIM];
            for (int j = 0; j < INPUT_DIM; j++) {
                input[j] = X[i][j];
            }
            
            double prediction = forward(input);
            double error = prediction - Y[i];
            total_loss += error * error;
        }
        
        return total_loss / n_samples;
    }
    

};

//se cargan los datos del archivo txt
void loadDataset(const string& filename, vector<vector<double>>& X, vector<double>& Y) {
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: No se pudo abrir el archivo de datos" << endl;
        return;
    }
    
    double x1, x2, x3, y;
    while (file >> x1 >> x2 >> x3 >> y) {
        vector<double> input = {x1, x2, x3};
        X.push_back(input);
        Y.push_back(y);
    }
    
    file.close();
    cout << "Dataset cargado: " << X.size() << " muestras" << endl;
}

