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

// Train Type Definitions
#define STOCHASTIC_GRADIENT_DESCENT 0
#define GRADIENT_DESCENT 1
#define RESISTANT_PROPAGATION 2
#define NATURAL_SELECTION 3
#define RANDOM_METHOD 4
#define LEVENBERG_MARQUARDT 5

// Network
class NeuralNetwork{
public:
    // Classes
    static struct standardTrainConfig {
        // General
        int trainType = GRADIENT_DESCENT;
        int epochs = 1000;

        vector<vector<float>> trainInputs;
        vector<vector<float>> trainOutputs;

        bool useWeightDecay = false;
        float weightDecayMultiplier = 0.9f;

        bool useDropout = false;
        int nodeBiasDropoutProbability = 10; // 1 in 10 (0.1)

        // Gradient Descent
        float learningRate = 1.0f;
        float momentum = 0.25f;
        bool useCyclicalLearningRateAndMomentum = false;

        // Stochastic Gradient Descent
        int entireBatchEpochIntervals = 500; // Every x epochs, the entire dataset is run
        int batchSize = 50;

        // Resistant Propagation
        float rpropWeightDecreaseMultiplier = 0.5f;
        float rpropWeightIncreaseMultiplier = 1.2f;

        // Levenberg Marquardt
        float dampingParameter = 0.01f;
        float dampIncreaseMultiplierLM = 10.0f;
        float dampDecreaseMultiplierLM = 0.1f;

        // Natural Selection
        int population = 10;
        float initialVariation = 50.0f;

        // Random Method
        float errorThreshold = 500.0f;
    } bar;

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
    vector<float> train(standardTrainConfig trainConfig);

    float activate(float x);
    float derivative(float x);

    void feedForward(vector<float> inputs);
    vector<float> predict(vector<float> inputs);
    void runTests(vector<vector<float>> inputs);

    void calculateDerivatives(vector<float> outputErrors, float errorMultiplier);
    void resetDerivativesAndResults();

    void decayWeights(float multiplier);

    // Dropout
    void randomlyDropNodes(int probability);
    void reactivateNodes();

    // Gradient Descent (updates after each example)
    void adjustWeightsGradientDescent(float lr, float momentum);
    vector<float> trainGradientDescent(standardTrainConfig trainConfig);

    // Stochastic Gradient Descent (select a few random train inputs)
    vector<float> trainStochasticGradientDescent(standardTrainConfig trainConfig);

    // Resistant Propagation
    vector<float> trainResistantPropagation(standardTrainConfig trainConfig);
    void adjustWeightsRPROP(float increase, float decrease, bool initialUpdate);
    
    // Natural Selection
    vector<vector<Node>> randomNodeWeights(vector<vector<Node>> initial, float variation);
    vector<vector<Bias>> randomBiasWeights(vector<vector<Bias>> initial, float variation);
    vector<float> trainNaturalSelectionMethod(standardTrainConfig trainConfig);

    // Random Weights Method
    void randomizeWeights();
    vector<float> trainRandomMethod(standardTrainConfig trainConfig);

    // Levenberg Marquardt
    void addDeltasLM(vector<float> deltas);
    vector<float> calculateDeltasLM(vector<vector<float>> jacobianMatrix, vector<vector<float>> costMatrix, float dampen);
    vector<float> getJacobianRowLM();
    vector<float> trainLevenbergMarquardt(standardTrainConfig trainConfig); 
};

#endif // !NEURALNETWORK_H