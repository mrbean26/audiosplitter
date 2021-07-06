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

struct audioFileConfig;

class NeuralNetwork{
public:
    // Classes
    static struct standardTrainConfig {
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

        float dampingParameter = 0.001f;
        float dampIncreaseMultiplierLM = 2.0f;
        float dampDecreaseMultiplierLM = 0.5f;
    };

    static struct Node {
        float value = 0.0f;

        float derivativeErrorValue = 0.0f;
        vector<float> outWeights;
        vector<float> previousDeltas;

        bool active = true;
    };

    static struct Bias {
        vector<float> outWeights;
        vector<float> previousDeltas;

        bool active = true;
    };

    // Config
    vector<vector<Node>> layerNodes;
    vector<vector<Bias>> layerBiases;

    string activationType = "sigmoid";
    int layerCount = 0;

    // Initialising
    NeuralNetwork(vector<int> layers, vector<int> biases, string activation);
    void initialiseWeights();

    void saveWeightsToFile(string directory);
    void loadWeightsFromFile(string directory);

    // General Training
    float activate(float x);
    float derivative(float x);

    void feedForward(vector<float> inputs);
    vector<float> predict(vector<float> inputs);
    void runTests(vector<vector<float>> inputs);

    void calculateDerivatives(vector<float> outputErrors, float errorMultiplier);
    void resetDerivativesAndResults();

    void decayWeights(float multiplier);

    static void trainSeveralConfigurations(audioFileConfig config, vector<vector<float>> inputSet, vector<vector<float>> outputSet, int epochs, int minimumLayerCount, int iterationsPerEach, int lowestLayerSize, float lowestLearningRate, float lowestMomentum);

    // Dropout
    void randomlyDropNodes(int probability);
    void reactivateNodes();

    // Gradient Descent
    void adjustWeights(float lr, float momentum);
    vector<float> train(standardTrainConfig trainConfig);
    
    // Resistant Propagation
    vector<float> trainResistantPropagation(standardTrainConfig trainConfig);
    void adjustWeightsRPROP(float increase, float decrease);
    
    // Natural Selection
    vector<vector<Node>> randomNodeWeights(vector<vector<Node>> initial, float variation);
    vector<vector<Bias>> randomBiasWeights(vector<vector<Bias>> initial, float variation);
    vector<float> trainNaturalSelectionMethod(vector<vector<float>> trainInputs, vector<vector<float>> trainOutputs, int epochs, int population, float initialVariation);

    // Random Weights Method
    void randomizeWeights();
    void trainRandomMethod(int epochs, float errorThreshold, vector<vector<float>> trainInputs, vector<vector<float>> trainOutputs);

    // Levenberg Marquardt
    void addDeltasLM(vector<float> deltas);
    vector<float> calculateDeltasLM(float cost, float dampen);
    vector<float> trainLevenbergMarquardt(standardTrainConfig trainConfig); 
};

#endif // !NEURALNETWORK_H