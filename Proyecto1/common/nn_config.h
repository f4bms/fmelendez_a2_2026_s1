#pragma once

// Configuración base compartida para todas las variantes.
// Se mantiene como constantes globales para minimizar cambios en el resto del código.

constexpr int INPUT_DIM = 3;
constexpr int HIDDEN_NEURONS = 30;
constexpr int OUTPUT_DIM = 1;

constexpr double LEARNING_RATE = 0.08;
constexpr int EPOCHS = 1500;
