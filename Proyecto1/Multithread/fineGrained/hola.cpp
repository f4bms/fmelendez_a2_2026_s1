#include <random>
#include <iostream>

int main() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 100);

    for (int i = 0; i < 5; i++) {
        std::cout << dist(rng) << std::endl;
    }
}