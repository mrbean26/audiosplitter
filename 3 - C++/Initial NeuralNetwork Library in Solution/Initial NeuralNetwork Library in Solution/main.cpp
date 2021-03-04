#include "Headers/NeuralNetwork.h"
#include <time.h>

int main(){
    srand(time(NULL));
    NeuralNetwork net = NeuralNetwork({2, 2, 1}, {1, 1, 0}, "tanh");

    vector<vector<float>> inputs = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};
    vector<vector<float>> outputs = {{0.0f}, {1.0f}, {1.0f}, {0.0f}};

    net.train(inputs, outputs, 100000, 0.1f, 0.5f);
    net.runTests(inputs);

    system("pause");
    return 0;
}
