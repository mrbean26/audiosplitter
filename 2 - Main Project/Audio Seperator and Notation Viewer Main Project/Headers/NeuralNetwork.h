#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

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

struct standardTrainConfig {
    vector<vector<float>> trainInputs;
    vector<vector<float>> trainOutputs;

    int epochs = 1000;

    float learningRate = 1.0f;
    float momentum = 0.25f;

    bool useCyclicalLearningRateAndMomentum = false;

    bool useWeightDecay = false;
    float weightDecayMultiplier = 0.9f;

    float rpropWeightDecreaseMultiplier = 0.5f;
    float rpropWeightIncreaseMultiplier = 1.2f;

    bool useDropout = false;
    int nodeBiasDropoutProbability = 10; // 1 in 10 (0.1)
};

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

    bool active = true;
};

struct Bias{
    vector<float> outWeights;
    vector<float> previousDeltas;

    bool active = true;
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

    void randomlyDropNodes(int probability);
    void reactivateNodes();

    void calculateDerivatives(vector<float> outputErrors);
    void decayWeights(float multiplier);
    void adjustWeights(float lr, float momentum);
    void resetDerivativesAndResults();

    void randomizeWeights();
    void trainRandomMethod(int epochs, float errorThreshold, vector<vector<float>> trainInputs, vector<vector<float>> trainOutputs);

    vector<vector<Node>> randomNodeWeights(vector<vector<Node>> initial, float variation);
    vector<vector<Bias>> randomBiasWeights(vector<vector<Bias>> initial, float variation);
    vector<float> trainNaturalSelectionMethod(vector<vector<float>> trainInputs, vector<vector<float>> trainOutputs, int epochs, int population, float initialVariation);

    vector<float> trainResistantPropagation(standardTrainConfig trainConfig);
    void adjustWeightsRPROP(float increase, float decrease);

    vector<float> train(standardTrainConfig trainConfig);
    void runTests(vector<vector<float>> inputs);

    void saveWeightsToFile(string directory);
    void loadWeightsFromFile(string directory);
};

#endif // !NEURALNETWORK_H