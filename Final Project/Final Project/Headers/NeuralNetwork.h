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

#include "audio.h"
using namespace std;

string vectorToString(vector<float> used);

#define SIGMOIDAL_STEP_FUNCTION_MULTIPLIER 5

// Activations
#define SIGMOID 0
#define TANH 1
#define RELU 2
#define LEAKY_RELU 3
#define STEP 4
#define SOFTMAX 5

// Train Type Definitions
#define STOCHASTIC_GRADIENT_DESCENT 0
#define GRADIENT_DESCENT 1
#define RESISTANT_PROPAGATION 2
#define RANDOM_METHOD 3
#define LEVENBERG_MARQUARDT 4
#define BATCH_GRADIENT_DESCENT 5

// Learning Rate Calculation Definition
#define FIXED_LEARNING_RATE 0
#define CYCLICAL_LEARNING_RATE 1
#define ADAM_LEARNING_RATE 2
#define DECREASING_LEARNING_RATE 3

// Fitness Function Options for Natural Selection
#define ABSOLUTE_ERROR 0
#define SQUARED_ERROR 1
#define ROOT_SQUARED_ERROR 2
#define MEAN_SQUARED_ERROR 3

// Parent Selection Methods
#define TOP_PARENTS 0 
#define EXPONENTIAL_PARENTS 1
#define PROBABILITY_PARENTS 2

// Parent Reproduction Methods
#define AVERAGE_PARENTS 0
#define WEIGHTED_PARENTS 1

// Architechture Selection
#define ACTIVATION_NONLINEAR_ONLY 0
#define ACTIVATION_ALL 1
#define ACTIVATION_SIGMOID_ONLY 2

// Network
class NeuralNetwork{
public:
    // Classes
    static struct gradientDescentConfig {
        int learningRateType = FIXED_LEARNING_RATE;

        // "ADAM" LR Parameters
        float betaOne = 0.9f;
        float betaTwo = 0.999f;
        float epsillon = 0.00000001f;

        // Stochastic Gradient Descent
        int entireBatchEpochIntervals = 500; // Every x epochs, the entire dataset is run
        int batchSize = 50;

        int datasetRefreshInterval = 5; // Refresh dataset every x epochs
        audioFileConfig datasetAudioConfig; // Used for generating dataset
        
        bool useAllSongDataset = false;
        int allSongDatasetStart = 0;
        int allSongDatasetEnd = 100;

        // batch descent
        bool useThreading = false;
    } gradientDescent;
    static struct resistantPropagationConfig {
        float rpropWeightDecreaseMultiplier = 0.5f;
        float rpropWeightIncreaseMultiplier = 1.2f;
    } resistantPropagation;
    static struct levenbergMarquardtConfig {
        float dampingParameter = 0.01f;
        float dampIncreaseMultiplierLM = 10.0f;
        float dampDecreaseMultiplierLM = 0.1f;
    } levenbergMarquardt;
    static struct naturalSelectionConfig {
        int population = 10;
        int parentCount = 2;

        float lowestInitialisedWeight = -10.0f;
        float highestInitialisedWeight = 10.0f;

        int fitnessFunctionType = ABSOLUTE_ERROR;
        int parentSelectionMethod = TOP_PARENTS;

        int breedingMethod = WEIGHTED_PARENTS;
        bool useChildMutation = true;

        bool useStochasticDataset = false;
        int stochasticDatasetSize = 200;
        bool useThreading = true;

        // Architechture Selection
        int selectionAllowedActivations = ACTIVATION_NONLINEAR_ONLY;

        int selectionConvergenceCounter = 10;
        float selectionConvergenceValue = 1.0f;

        int selectionMinLayers = 3;
        int selectionMaxLayers = 18;

        int selectionMinNodes = 1;
        int selectionMaxNodes = 640;

        int selectionMinBias = 0;
        int selectionMaxBias = 3;
    } naturalSelection;
    static struct randomMethodConfig {
        float errorThreshold = 500.0f;
    };

    static struct standardTrainConfig {
        // General
        int trainType = GRADIENT_DESCENT;
        int epochs = 1000;

        float learningRate = 1.0f;
        float momentum = 0.25f;

        vector<vector<float>> trainInputs;
        vector<vector<float>> trainOutputs;

        bool useWeightDecay = false;
        float weightDecayMultiplier = 0.9f;

        bool useDropout = false;
        int nodeBiasDropoutProbability = 10; // 1 in 10 (0.1)

        gradientDescentConfig gradientDescent;
        resistantPropagationConfig resistantPropagation;
        levenbergMarquardtConfig levenbergMarquardt;
        naturalSelectionConfig naturalSelection;
        randomMethodConfig randomMethod;
    } trainConfig;

    static struct Node {
        // Runtime
        float value = 0.0f;
        float derivativeErrorValue = 0.0f;
        vector<float> outWeights;

        // Momentum
        vector<float> previousDeltas;

        // Dropout
        bool active = true;

        // ADAM Learning Rates
        vector<float> previousExponentials;
        vector<float> previousSquaredExponentials;

        // Batch Graient Descent
        vector<float> accumulativeDeltas;
    };
    static struct Bias {
        // Runtime
        vector<float> outWeights;

        // Momentum
        vector<float> previousDeltas;

        // Dropout
        bool active = true;

        // ADAM Learning Rates
        vector<float> previousExponentials;
        vector<float> previousSquaredExponentials;

        // Batch Graient Descent
        vector<float> accumulativeDeltas;
    };

    // Config
    vector<vector<Node>> layerNodes;
    vector<vector<Bias>> layerBiases;

    vector<int> activations;
    int layerCount = 0;

    // Initialising
    NeuralNetwork(); // Default Constructor
    NeuralNetwork(vector<int> layers, vector<int> biases, vector<int> activationLayers);
    void initialiseWeights();

    void saveWeightsToFile(string directory);
    void loadWeightsFromFile(string directory);

    // General Training
    void setupNetworkForTraining(standardTrainConfig trainConfig);
    vector<float> train(standardTrainConfig trainConfig);

    float activate(float x, int layer);
    float derivative(float x, int layer);

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
    void adjustWeightsADAM(standardTrainConfig trainConfig);
    vector<float> trainGradientDescent(standardTrainConfig trainConfig);

    pair<float, float> calculateLRMomentum(int epoch, standardTrainConfig trainConfig);
};

#endif // !NEURALNETWORK_H