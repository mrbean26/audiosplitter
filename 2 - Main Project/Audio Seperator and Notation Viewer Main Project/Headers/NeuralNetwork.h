#include <iostream>
#include <vector>

#include <string>
#include <fstream>
#include <sstream>

#include <algorithm>
#include <math.h>
#include <stdlib.h>

using namespace std;

string vectorToString(vector<float> used);

namespace activations{
    float sigmoid(float x);
    float tanh(float x);
    float relu(float x);
    float leaky_relu(float x);

    namespace derivatives{
        float sigmoid(float x);
        float tanh(float x);
        float relu(float x);
        float leaky_relu(float x);
    }
}

struct Node{
    float value = 0.0f;

    float derivativeErrorValue = 0.0f;
    vector<float> outWeights;
    vector<float> previousDeltas;
};

struct Bias{
    vector<float> outWeights;
    vector<float> previousDeltas;
};

class NeuralNetwork{
public:
    float activate(float x);
    float derivative(float x);

    vector<vector<Node>> layerNodes;
    vector<vector<Bias>> layerBiases;

    string activationType = "sigmoid";
    int layerCount = 0;

    NeuralNetwork(vector<int> layers, vector<int> biases, string activation);
    void initialiseWeights();

    void feedForward(vector<float> inputs);
    vector<float> predict(vector<float> inputs);

    void calculateDerivatives(vector<float> outputErrors);
    void adjustWeights(float lr, float momentum);
    void resetDerivativesAndResults();

    void randomizeWeights();
    void trainRandomMethod(int epochs, float errorThreshold, vector<vector<float>> trainInputs, vector<vector<float>> trainOutputs);

    vector<float> train(vector<vector<float>> trainInputs, vector<vector<float>> trainOutputs,
        int epochs, float lr, float momentum);
    void runTests(vector<vector<float>> inputs);

    void saveWeightsToFile(string directory);
    void loadWeightsFromFile(string directory);
};
