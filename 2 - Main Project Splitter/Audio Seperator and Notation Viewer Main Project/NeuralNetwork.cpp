#include "Headers/NeuralNetwork.h"

#include <fstream>
#include <sstream>

#include <random>

#include <thread>
#include <future>

#include "Headers/files.h"
#include "Headers/matrices.h"

// Initialisation Functions
NeuralNetwork::NeuralNetwork() {}
NeuralNetwork::NeuralNetwork(vector<int> layers, vector<int> biases, vector<int> activationLayers) {
    layerCount = layers.size();
    for (int i = 0; i < layerCount; i++) {
        vector<Node> newLayer;
        for (int n = 0; n < layers[i]; n++) {
            newLayer.push_back(Node());
        }
        layerNodes.push_back(newLayer);

        vector<Bias> newBiasLayer;
        for (int b = 0; b < biases[i]; b++) {
            newBiasLayer.push_back(Bias());
        }
        layerBiases.push_back(newBiasLayer);
    }
    initialiseWeights();

    activations = activationLayers;
}

// random functions for initialising weights
float randomMinimum = -1.0f;
float randomMaximum = 1.0f;
float randomFloat() {
    float result = randomMinimum + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (randomMaximum - randomMinimum)));
    return result;
}

// small random value generation for causing mutation in natural selection algorithms
default_random_engine generator;
normal_distribution<float> distribution(0.0f, 0.03f);
float normalDistribution() {
    return distribution(generator);
}

// generation of random values for generating network architechtures
normal_distribution<float> distributionArchitechture(0.0f, 0.75f);
float architechtureNormalDistribution() {
    return distributionArchitechture(generator);
}

// clear weight vectors and initialise random values
void NeuralNetwork::initialiseWeights() {
    // clear weights
    for (int i = 0; i < layerCount; i++) {
        int currentNodeCount = layerNodes[i].size();
        for (int n = 0; n < currentNodeCount; n++) {
            layerNodes[i][n].outWeights.clear();
        }

        int currentBiasCount = layerBiases[i].size();
        for (int b = 0; b < currentBiasCount; b++) {
            layerBiases[i][b].outWeights.clear();
        }
    }

    // initialise
    for (int i = 0; i < layerCount - 1; i++) {
        int currentNodeCount = layerNodes[i].size();
        int nextNodeCount = layerNodes[i + 1].size();

        // Initialise Weights with Random Values
        for (int n = 0; n < currentNodeCount; n++) {
            for (int n1 = 0; n1 < nextNodeCount; n1++) {
                layerNodes[i][n].outWeights.push_back(randomFloat());
            }
        }

        int currentBiasCount = layerBiases[i].size();
        for (int b = 0; b < currentBiasCount; b++) {
            for (int n = 0; n < nextNodeCount; n++) {
                layerBiases[i][b].outWeights.push_back(randomFloat());
            }
        }
    }
}

// save binary weight values for nodes & bias values
void NeuralNetwork::saveWeightsToFile(string directory) {
    string fileNameNodes = directory + "nodeWeights.bin";
    string fileNameBias = directory + "biasWeights.bin";

    // Push All Node Weights into Vector in Order of Nodes then Layers
    vector<float> allWeightsNode;
    for (int layerNum = 0; layerNum < layerCount; layerNum++) {
        int nodeCount = layerNodes[layerNum].size();

        for (int node = 0; node < nodeCount; node++) {
            int outCount = layerNodes[layerNum][node].outWeights.size();
            for (int weight = 0; weight < outCount; weight++) {
                allWeightsNode.push_back(layerNodes[layerNum][node].outWeights[weight]);
            }
        }
    }

    // Push All Bias Weights into Vector in Order of Nodes then Layers
    vector<float> allWeightsBias;
    for (int layerNum = 0; layerNum < layerCount; layerNum++) {
        int nodeCount = layerBiases[layerNum].size();

        for (int node = 0; node < nodeCount; node++) {
            int outCount = layerBiases[layerNum][node].outWeights.size();
            for (int weight = 0; weight < outCount; weight++) {
                allWeightsBias.push_back(layerBiases[layerNum][node].outWeights[weight]);
            }
        }
    }

    // Write Vectors to file
    ofstream outputNodes;
    outputNodes.open(fileNameNodes, ios::out | ios::binary);
    outputNodes.write(reinterpret_cast<char*>(&allWeightsNode[0]), allWeightsNode.size() * sizeof(float));
    outputNodes.close();

    ofstream outputBias;
    outputBias.open(fileNameBias, ios::out | ios::binary);
    outputBias.write(reinterpret_cast<char*>(&allWeightsBias[0]), allWeightsBias.size() * sizeof(float));
    outputBias.close();
}
void NeuralNetwork::loadWeightsFromFile(string directory) {
    // Calculate Total Network Parameters
    string fileNameNodes = directory + "nodeWeights.bin";
    string fileNameBias = directory + "biasWeights.bin";

    int totalNodeWeightCount = 0;
    for (int i = 0; i < layerCount - 1; i++) {
        totalNodeWeightCount += layerNodes[i].size() * layerNodes[i + 1].size();;
    }

    int totalBiasWeightCount = 0;
    for (int i = 0; i < layerCount - 1; i++) {
        totalBiasWeightCount += layerBiases[i].size() * layerNodes[i + 1].size();
    }

    vector<float> allNodeWeights(totalNodeWeightCount);
    vector<float> allBiasWeights(totalBiasWeightCount);

    // Open Files Containing Weights, Written in Binary to Save much more space (8 bits per float instead of around 56)
    ifstream inputNodes;
    inputNodes.open(fileNameNodes, ios::in | ios::binary);
    inputNodes.read(reinterpret_cast<char*>(&allNodeWeights[0]), totalNodeWeightCount * sizeof(float));
    inputNodes.close();

    ifstream inputBiases;
    inputBiases.open(fileNameBias, ios::in | ios::binary);
    inputBiases.read(reinterpret_cast<char*>(&allBiasWeights[0]), totalBiasWeightCount * sizeof(float));
    inputBiases.close();

    // Load Node Weights In Order of Layer Nodes, Then Next Layer
    int layerNum = 0;
    int nodeNum = 0;
    int weightNum = 0;

    for (int i = 0; i < totalNodeWeightCount; i++) {
        layerNodes[layerNum][nodeNum].outWeights[weightNum] = allNodeWeights[i];

        weightNum += 1;
        if (weightNum == layerNodes[layerNum][nodeNum].outWeights.size()) {
            weightNum = 0;
            nodeNum += 1;

            if (nodeNum == layerNodes[layerNum].size()) {
                nodeNum = 0;
                layerNum += 1;
            }
        }
    }

    // Load Bias Weights In Order of Layer Nodes, Then Next Layer
    layerNum = 0;
    nodeNum = 0;
    weightNum = 0;

    for (int i = 0; i < totalBiasWeightCount; i++) {
        layerBiases[layerNum][nodeNum].outWeights[weightNum] = allBiasWeights[i];

        weightNum += 1;
        if (weightNum == layerBiases[layerNum][nodeNum].outWeights.size()) {
            weightNum = 0;
            nodeNum += 1;

            if (nodeNum == layerBiases[layerNum].size()) {
                nodeNum = 0;
                layerNum += 1;
            }
        }
    }
}

// General Training
void NeuralNetwork::setupNetworkForTraining(standardTrainConfig trainConfig) {
    for (int i = 0; i < layerCount - 1; i++) {
        int nodeCount = layerNodes[i].size();
        int biasCount = layerBiases[i].size();

        int weightCount = layerNodes[i + 1].size();
        vector<float> emptyVector(weightCount);

        // Only add vectors for fields that will be used - this is decided on which learning method is chosen
        for (int n = 0; n < nodeCount; n++) {
            if (trainConfig.gradientDescent.learningRateType == ADAM_LEARNING_RATE) {
                layerNodes[i][n].previousExponentials = emptyVector;
                layerNodes[i][n].previousSquaredExponentials = emptyVector;
            }
            else {
                layerNodes[i][n].previousDeltas = emptyVector;
            }
            if (trainConfig.trainType == BATCH_GRADIENT_DESCENT) {
                layerNodes[i][n].accumulativeDeltas = emptyVector;
            }
        }
        for (int b = 0; b < biasCount; b++) {
            if (trainConfig.gradientDescent.learningRateType == ADAM_LEARNING_RATE) {
                layerBiases[i][b].previousExponentials = emptyVector;
                layerBiases[i][b].previousSquaredExponentials = emptyVector;
            }
            else {
                layerBiases[i][b].previousDeltas = emptyVector;
            }
            if (trainConfig.trainType == BATCH_GRADIENT_DESCENT) {
                layerBiases[i][b].accumulativeDeltas = emptyVector;
            }
        }
    }
}
vector<float> NeuralNetwork::train(standardTrainConfig trainConfig) {
    vector<float> result;
    setupNetworkForTraining(trainConfig);
    
    if (trainConfig.trainType == STOCHASTIC_GRADIENT_DESCENT) {
        result = trainStochasticGradientDescent(trainConfig);
    }
    if (trainConfig.trainType == GRADIENT_DESCENT) {
        result = trainGradientDescent(trainConfig);
    }
    if (trainConfig.trainType == RESISTANT_PROPAGATION) {
        result = trainResistantPropagation(trainConfig);
    }
    if (trainConfig.trainType == RANDOM_METHOD) {
        result = trainRandomMethod(trainConfig);
    }
    if (trainConfig.trainType == LEVENBERG_MARQUARDT) {
        result = trainLevenbergMarquardt(trainConfig);
    }
    if (trainConfig.trainType == BATCH_GRADIENT_DESCENT) {
        result = trainBatchGradientDescent(trainConfig);
    }

    return result;
}

// Activation Functions - the step function is not great due to extreme gradients
float layerSoftmaxTotal = 0.0f;
float NeuralNetwork::activate(float x, int layer) {
    if (activations[layer] == SIGMOID) {
        return 1.0f / (1.0f + exp(-x));
    }
    if (activations[layer] == TANH) {
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    }
    if (activations[layer] == RELU) {
        if (x > 0.0f) {
            return x;
        }
        return 0.0f;
    }
    if (activations[layer] == LEAKY_RELU) {
        if (x > 0.01f * x) {
            return x;
        }
        return x * 0.01f;
    }
    if (activations[layer] == STEP) {
        return 1.0f / (1.0f + exp(-SIGMOIDAL_STEP_FUNCTION_MULTIPLIER * x));
    }
    if (activations[layer] == SOFTMAX) {
        return expf(x) / layerSoftmaxTotal;
    }
    return 1.0f / (1.0f + exp(-x));
}
float NeuralNetwork::derivative(float x, int layer) {
    if (activations[layer] == SIGMOID) {
        return x * (1 - x);
    }
    if (activations[layer] == TANH) {
        return 1 - (x * x);
    }
    if (activations[layer] == RELU) {
        if (x > 0.0f) {
            return 1.0f;
        }
        return 0.0f;
    }
    if (activations[layer] == LEAKY_RELU) {
        if (x > 0.0f) {
            return 1.0f;
        }
        return 0.01f;
    }
    if (activations[layer] == STEP) {
        return x * x * SIGMOIDAL_STEP_FUNCTION_MULTIPLIER * ((1.0f / x) - 1.0f);
    }
    if (activations[layer] == SOFTMAX) {
        return x * (1.0f - x);
    }
    return x * (1 - x);
}

// Feeding values forward through algorithm
void NeuralNetwork::feedForward(vector<float> inputs) {
    // Set Accumulative Values to 0
    resetDerivativesAndResults();

    // Set First Layer (unactivated) Nodes to Input Values
    int firstLayerCount = layerNodes[0].size();
    for (int i = 0; i < firstLayerCount; i++) {
        layerNodes[0][i].value = inputs[i];
    }

    // Feed Forward
    for (int i = 0; i < layerCount; i++) {
        int thisLayerCount = layerNodes[i].size();
        
        int outWeightCount = 0;
        if (i < layerCount - 1) {
            outWeightCount = layerNodes[i + 1].size();
        }

        // After Weighted Sum, Pass node value through activation function
        if (i > 0) {
            if (activations[i] == SOFTMAX) {
                layerSoftmaxTotal = 0.0f;

                for (int n = 0; n < thisLayerCount; n++) {
                    layerSoftmaxTotal = layerSoftmaxTotal + expf(layerNodes[i][n].value);
                }
            }

            for (int n = 0; n < thisLayerCount; n++) {
                layerNodes[i][n].value = activate(layerNodes[i][n].value, i);
            }
        }

        // Add Weighted Sum To Nodes in Next Layers
        float currentNodeValue;
        for (int n = 0; n < thisLayerCount; n++) {
            if (!layerNodes[i][n].active) {
                continue;
            }
            currentNodeValue = layerNodes[i][n].value;

            for (int w = 0; w < outWeightCount; w++) {
                layerNodes[i + 1][w].value += currentNodeValue * layerNodes[i][n].outWeights[w];
            }
        }

        // Add Bias Weights (useful for when 0 values are present)
        int thisBiasCount = layerBiases[i].size();
        for (int b = 0; b < thisBiasCount; b++) {
            if (!layerBiases[i][b].active) {
                continue;
            }

            for (int w = 0; w < outWeightCount; w++) {
                layerNodes[i + 1][w].value += 1.0f * layerBiases[i][b].outWeights[w];
            }
        }
    }
}
vector<float> NeuralNetwork::predict(vector<float> inputs) {
    feedForward(inputs);

    vector<float> result;
    int lastLayerSize = layerNodes[layerCount - 1].size();

    for (int i = 0; i < lastLayerSize; i++) {
        result.push_back(layerNodes[layerCount - 1][i].value);
    }

    return result;
}
void NeuralNetwork::runTests(vector<vector<float>> inputs) {
    int count = inputs.size();
    for (int i = 0; i < count; i++) {
        cout << vectorToString(predict(inputs[i])) << endl;
    }
}

// calculating derivatives of each weight with respect to error
void NeuralNetwork::calculateDerivatives(vector<float> outputErrors, float errorMultiplier = 1.0f) {
    // with outputErrors as actual - target
    int finalLayerCount = layerNodes[layerCount - 1].size();
    
    for (int i = 0; i < finalLayerCount; i++) {
        layerNodes[layerCount - 1][i].derivativeErrorValue = derivative(layerNodes[layerCount - 1][i].value, layerCount - 1) * outputErrors[i] * errorMultiplier;
    }
    
    // Backpropagate by Calculating Partial Derivatives of Each Node with Respect to The Error
    for (int i = layerCount - 2; i > -1; i--) {
        int currentLayerCount = layerNodes[i].size();
        int outWeightCount = layerNodes[i + 1].size();

        for (int n = 0; n < currentLayerCount; n++) {
            if (!layerNodes[i][n].active) {
                continue;
            }
            
            float valueMultiplier = derivative(layerNodes[i][n].value, i);
            if (i == 0) {
                valueMultiplier = layerNodes[i][n].value; // This value is used for input layer due to this layer not being activated
            }

            for (int w = 0; w < outWeightCount; w++) {
                if (!layerNodes[i + 1][w].active) {
                    continue;
                }

                // Derivative is Accumulative to Nodes in Next Layer
                layerNodes[i][n].derivativeErrorValue += valueMultiplier * layerNodes[i][n].outWeights[w] * layerNodes[i + 1][w].derivativeErrorValue;                
            }
        }
    }
}
void NeuralNetwork::resetDerivativesAndResults() {
    for (int i = 0; i < layerCount; i++) {
        int nodeCount = layerNodes[i].size();

        for (int n = 0; n < nodeCount; n++) {
            layerNodes[i][n].derivativeErrorValue = 0.0f;
            layerNodes[i][n].value = 0.0f;
        }
    }
}

// Dropout & Weight decay to help prevent overfitting to a training dataset
void NeuralNetwork::decayWeights(float multiplier) {
    // decaying weights helps in preventing overfitting dataset

    for (int i = 0; i < layerCount - 1; i++) {
        int nodeCount = layerNodes[i].size();
        int biasCount = layerBiases[i].size();

        int weightCount = layerNodes[i + 1].size();

        for (int n = 0; n < nodeCount; n++) {
            for (int w = 0; w < weightCount; w++) {
                layerNodes[i][n].outWeights[w] *= multiplier;
            }
        }

        for (int n = 0; n < biasCount; n++) {
            for (int w = 0; w < weightCount; w++) {
                layerBiases[i][n].outWeights[w] *= multiplier;
            }
        }
    }
}
void NeuralNetwork::randomlyDropNodes(int probability) {
    // Randomly Drop Nodes except for output layer
    for (int i = 0; i < layerCount - 1; i++) {
        int nodeCount = layerNodes[i].size();
        int biasCount = layerBiases[i].size();

        int droppedNodes = 0;
        for (int n = 0; n < nodeCount; n++) {
            if (rand() % probability == 1) {
                layerNodes[i][n].active = false;
                droppedNodes = droppedNodes + 1;
            }
        }

        for (int b = 0; b < biasCount; b++) {
            if (rand() % probability == 1) {
                layerBiases[i][b].active = false;
            }
        }

        // Check at least one node open in layer, if not then free up random node
        if (droppedNodes == nodeCount) {
            int index = rand() % nodeCount;
            layerNodes[i][index].active = true;
        }
    }
}
void NeuralNetwork::reactivateNodes() {
    for (int i = 0; i < layerCount - 1; i++) {
        int nodeCount = layerNodes[i].size();
        int biasCount = layerBiases[i].size();

        for (int n = 0; n < nodeCount; n++) {
            layerNodes[i][n].active = true;
        }

        for (int b = 0; b < biasCount; b++) {
            layerBiases[i][b].active = true;
        }
    }
}

// Gradient Descent (updates after each example)
void NeuralNetwork::adjustWeightsGradientDescent(float lr, float momentum) {
    for (int i = 0; i < layerCount; i++) {
        int nodeCount = layerNodes[i].size();

        // Adjust Weights That Come From Nodes
        for (int n = 0; n < nodeCount; n++) {
            if (!layerNodes[i][n].active) {
                continue;
            }

            int weightCount = layerNodes[i][n].outWeights.size();
            for (int w = 0; w < weightCount; w++) {
                // Calculate Gradient of Error With Respect To Node
                float newDelta = layerNodes[i][n].value * layerNodes[i + 1][w].derivativeErrorValue * lr;
                layerNodes[i][n].outWeights[w] += newDelta;
                
                // Add a proportion of last updates' adjustments
                layerNodes[i][n].outWeights[w] += layerNodes[i][n].previousDeltas[w] * momentum;
                layerNodes[i][n].previousDeltas[w] = newDelta;
            }
        }

        // Adjust Weights That Come From Biases
        int biasCount = layerBiases[i].size();
        for (int b = 0; b < biasCount; b++) {
            if (!layerBiases[i][b].active) {
                continue;
            }

            int outWeightCount = layerBiases[i][b].outWeights.size();
            for (int w = 0; w < outWeightCount; w++) {
                // Calculate Gradient of Error With Respect To Node
                float newDelta = 1.0f * layerNodes[i + 1][w].derivativeErrorValue * lr;
                layerBiases[i][b].outWeights[w] += newDelta;
                
                // Add a proportion of last updates' adjustments
                layerBiases[i][b].outWeights[w] += layerBiases[i][b].previousDeltas[w] * momentum;
                layerBiases[i][b].previousDeltas[w] = newDelta;
            }
        }
    }
}
void NeuralNetwork::adjustWeightsADAM(standardTrainConfig trainConfig) {
    for (int i = 0; i < layerCount; i++) {
        int nodeCount = layerNodes[i].size();

        // Adjust Weights That Come From Nodes
        for (int n = 0; n < nodeCount; n++) {
            if (!layerNodes[i][n].active) {
                continue;
            }

            int weightCount = layerNodes[i][n].outWeights.size();
            for (int w = 0; w < weightCount; w++) {
                // Calculate Gradient of Error With Respect To Node
                float gradient = layerNodes[i][n].value * layerNodes[i + 1][w].derivativeErrorValue;

                // ADAM Parameters#
                float newExponential = trainConfig.gradientDescent.betaOne * layerNodes[i][n].previousExponentials[w] + (1 - trainConfig.gradientDescent.betaOne) * gradient;
                float newSquaredExponential = trainConfig.gradientDescent.betaTwo * layerNodes[i][n].previousSquaredExponentials[w] + (1 - trainConfig.gradientDescent.betaTwo) * powf(gradient, 2.0f);
                
                float currentExponential = newExponential / (1 - trainConfig.gradientDescent.betaOne);
                float currentSquaredExponential = newSquaredExponential / (1 - trainConfig.gradientDescent.betaTwo);
                
                float learningRate = trainConfig.learningRate * (currentExponential / (sqrtf(currentSquaredExponential) + trainConfig.gradientDescent.epsillon));
                float delta = learningRate * gradient;
                
                layerNodes[i][n].outWeights[w] -= delta;

                // Update ADAM Parameters
                layerNodes[i][n].previousExponentials[w] = newExponential;
                layerNodes[i][n].previousSquaredExponentials[w] = newSquaredExponential;
            }
        }

        // Adjust Weights That Come From Biases
        int biasCount = layerBiases[i].size();
        for (int b = 0; b < biasCount; b++) {
            if (!layerBiases[i][b].active) {
                continue;
            }

            int outWeightCount = layerBiases[i][b].outWeights.size();
            for (int w = 0; w < outWeightCount; w++) {
                // Calculate Gradient of Error With Respect To Node
                float gradient = 1.0f * layerNodes[i + 1][w].derivativeErrorValue;

                // ADAM Parameters
                float newExponential = trainConfig.gradientDescent.betaOne * layerBiases[i][b].previousExponentials[w] + (1 - trainConfig.gradientDescent.betaOne) * gradient;
                float newSquaredExponential = trainConfig.gradientDescent.betaTwo * layerBiases[i][b].previousSquaredExponentials[w] + (1 - trainConfig.gradientDescent.betaTwo) * powf(gradient, 2.0f);

                float currentExponential = newExponential / (1 - trainConfig.gradientDescent.betaOne);
                float currentSquaredExponential = newSquaredExponential / (1 - trainConfig.gradientDescent.betaTwo);

                float learningRate = trainConfig.learningRate * (currentExponential / (sqrtf(currentSquaredExponential) + trainConfig.gradientDescent.epsillon));
                float delta = learningRate * gradient;

                layerBiases[i][b].outWeights[w] -= delta;

                // Update ADAM Parameters
                layerBiases[i][b].previousExponentials[w] = newExponential;
                layerBiases[i][b].previousSquaredExponentials[w] = newSquaredExponential;
            }
        }
    }
}
vector<float> NeuralNetwork::trainGradientDescent(standardTrainConfig trainConfig) {
    int trainDataCount = trainConfig.trainInputs.size();
    int outputCount = trainConfig.trainOutputs[0].size();

    vector<int> trainIndexes;
    for (int i = 0; i < trainDataCount; i++) {
        trainIndexes.push_back(i);
    }

    vector<float> result;
    for (int epoch = 0; epoch < trainConfig.epochs; epoch++) {
        // Randomly Shuffle Dataset Indexes to Prevent Overfitting
        random_shuffle(trainIndexes.begin(), trainIndexes.end());

        // Calculate Current Learning Rate
        pair<float, float> trainingParameters = calculateLRMomentum(epoch, trainConfig);
        float currentLearningRate = trainingParameters.first;
        float currentMomentum = trainingParameters.second;

        // Randomly Disable Some Nodes to Prevent Overfitting
        if (trainConfig.useDropout) {
            randomlyDropNodes(trainConfig.nodeBiasDropoutProbability);
        }

        float totalError = 0.0f;
        for (int t = 0; t < trainDataCount; t++) {
            if (t % 250 == 0) {
                //cout << epoch + 1 << ":" << t + 1 << "/" << trainDataCount << endl;
            }

            int currentIndex = trainIndexes[t];
            vector<float> result = predict(trainConfig.trainInputs[currentIndex]);

            // Calculate Differences In Actual Output
            vector<float> errors;
            for (int e = 0; e < outputCount; e++) {
                totalError += abs(trainConfig.trainOutputs[currentIndex][e] - result[e]);
                errors.push_back(trainConfig.trainOutputs[currentIndex][e] - result[e]);
            }

            calculateDerivatives(errors);

            // Update Parameters
            if (trainConfig.gradientDescent.learningRateType == ADAM_LEARNING_RATE) {
                adjustWeightsADAM(trainConfig);
            }
            else {
                adjustWeightsGradientDescent(currentLearningRate, trainConfig.momentum);
            }

            // Lower Some Weights To Prevent Overfitting
            if (trainConfig.useWeightDecay) {
                decayWeights(trainConfig.weightDecayMultiplier);
            }
        }

        reactivateNodes();

        cout << "Epoch: " << epoch + 1 << " / " << trainConfig.epochs << ", Total error from epoch: " << totalError << ", Layers: " << layerCount << ", LR:" << currentLearningRate << endl;
        result.push_back(totalError);
    }

    return result;
}

// calculating learning rate & Momentum according to chosen type
pair<float, float> NeuralNetwork::calculateLRMomentum(int epoch, standardTrainConfig trainConfig) {
    float currentLearningRate = trainConfig.learningRate;
    float currentMomentum = trainConfig.momentum;
    
    if (trainConfig.gradientDescent.learningRateType == CYCLICAL_LEARNING_RATE) {
        // Peak in Middle - Use Linear Function
        double currentCoefficient = double(epoch + 1) / (double(trainConfig.epochs));
        float value = 1.0f - abs(2.0f * (currentCoefficient - 0.5f));

        currentLearningRate = value * trainConfig.learningRate;
        currentMomentum = (1 - value) * trainConfig.momentum;
    }

    if (trainConfig.gradientDescent.learningRateType == DECREASING_LEARNING_RATE) {
        float multiplier = float(epoch + 1) / float(trainConfig.epochs);
        currentLearningRate = (1.0f - multiplier) * trainConfig.learningRate;
        currentMomentum = multiplier * trainConfig.learningRate;
    }

    return make_pair(currentLearningRate, currentMomentum);
}

// Stochastic Gradient Descent (select a few random train inputs)
vector<float> NeuralNetwork::trainStochasticGradientDescent(standardTrainConfig trainConfig) {
    // Useful Integers Calculated Before Iteration
    vector<vector<float>> usedInputs;
    vector<vector<float>> usedOutputs;
    
    vector<float> resultantErrors;
    for (int epoch = 0; epoch < trainConfig.epochs; epoch++) {
        // Create Random Sectional Indexes for Stochastic Training Unless Epoch is Divisible by Full Dataset Test Parameter
        // Create Mini Dataset
        if (!trainConfig.gradientDescent.useAllSongDataset) {
            int miniBatchSize = trainConfig.trainInputs.size() / trainConfig.gradientDescent.batchSize;

            if ((epoch + 1) % trainConfig.gradientDescent.entireBatchEpochIntervals == 0) {
                // Use entire dataset every few epochs to test peformance
                usedInputs = trainConfig.trainInputs;
                usedOutputs = trainConfig.trainOutputs;
            }
            else {
                // create mini dataset
                for (int i = 0; i < trainConfig.gradientDescent.batchSize; i++) {
                    int newIndex = (i * miniBatchSize) + (rand() % miniBatchSize);

                    usedInputs.push_back(trainConfig.trainInputs[newIndex]);
                    usedOutputs.push_back(trainConfig.trainOutputs[newIndex]);
                }
            }
        }
        if (trainConfig.gradientDescent.useAllSongDataset) {
            // generate dataset from parts of all songs in the current avaliable data files
            if (epoch % trainConfig.gradientDescent.datasetRefreshInterval == 0) { // Time to refresh dataset
                int chunksPerSong = trainConfig.gradientDescent.batchSize / (trainConfig.gradientDescent.allSongDatasetEnd - trainConfig.gradientDescent.allSongDatasetStart); // / song count
                pair<vector<vector<float>>, vector<vector<float>>> allSongMiniDataset = generateAllSongDataSet(trainConfig.gradientDescent.datasetAudioConfig, chunksPerSong, trainConfig.gradientDescent.allSongDatasetStart, trainConfig.gradientDescent.allSongDatasetEnd);

                usedInputs = allSongMiniDataset.first;
                usedOutputs = allSongMiniDataset.second;
            }
        }

        int outputCount = usedOutputs[0].size();

        // Calculate Current Learning Rate
        pair<float, float> trainingParameters = calculateLRMomentum(epoch, trainConfig);
        float currentLearningRate = trainingParameters.first;
        float currentMomentum = trainingParameters.second;

        // Randomly Disable Some Nodes to Prevent Overfitting
        if (trainConfig.useDropout) {
            randomlyDropNodes(trainConfig.nodeBiasDropoutProbability);
        }

        // Adjust Parameters
        float totalError = 0.0f;
        for (int t = 0; t < usedInputs.size(); t++) {
            vector<float> result = predict(usedInputs[t]);

            // Calculate Differences In Actual Output
            vector<float> errors;
            for (int e = 0; e < outputCount; e++) {
                totalError += abs(usedOutputs[t][e] - result[e]);
                errors.push_back(usedOutputs[t][e] - result[e]);
            }
            
            calculateDerivatives(errors);

            // Update Parameters
            if (trainConfig.gradientDescent.learningRateType == ADAM_LEARNING_RATE) {
                adjustWeightsADAM(trainConfig);
            }
            else {
                adjustWeightsGradientDescent(currentLearningRate, trainConfig.momentum);
            }

            // Lower Some Weights To Prevent Overfitting
            if (trainConfig.useWeightDecay) {
                decayWeights(trainConfig.weightDecayMultiplier);
            }
        }

        // Reset Network
        reactivateNodes();

        cout << "Epoch: " << epoch + 1 << " / " << trainConfig.epochs << ", Approximate total error from epoch: " << totalError << ", Layers: " << layerCount << ", LR:" << currentLearningRate << endl;
        resultantErrors.push_back(totalError);
    }

    return resultantErrors;
}

// Resistant Propagation
vector<float> NeuralNetwork::trainResistantPropagation(standardTrainConfig trainConfig) {
    // Useful Integers Calculated Before Iteration
    int trainDataCount = trainConfig.trainInputs.size();
    int outputCount = trainConfig.trainOutputs[0].size();

    // All Possible Indexes Across Dataset
    vector<int> trainIndexes;
    for (int i = 0; i < trainDataCount; i++) {
        trainIndexes.push_back(i);
    }

    vector<float> result;
    for (int epoch = 0; epoch < trainConfig.epochs; epoch++) {
        // Randomly Shuffle Dataset Indexes to Prevent Overfitting
        random_shuffle(trainIndexes.begin(), trainIndexes.end());

        float totalError = 0.0f;
        for (int t = 0; t < trainDataCount; t++) {
            int currentIndex = trainIndexes[t];
            vector<float> result = predict(trainConfig.trainInputs[currentIndex]);

            // Calculate Differences In Actual Output
            vector<float> errors;
            for (int e = 0; e < outputCount; e++) {
                totalError += abs(trainConfig.trainOutputs[currentIndex][e] - result[e]);
                errors.push_back(trainConfig.trainOutputs[currentIndex][e] - result[e]);
            }

            // Update network parameters 
            calculateDerivatives(errors);
            adjustWeightsRPROP(trainConfig.resistantPropagation.rpropWeightIncreaseMultiplier, trainConfig.resistantPropagation.rpropWeightDecreaseMultiplier, epoch == 0 && t == 0);

            // Lower Some Weights To Prevent Overfitting
            if (trainConfig.useWeightDecay) {
                decayWeights(trainConfig.weightDecayMultiplier);
            }
        }

        cout << "Epoch: " << epoch + 1 << " / " << trainConfig.epochs << ", Total error from epoch: " << totalError << ", Layers: " << layerCount << endl;
        result.push_back(totalError);
    }

    return result;
}
void NeuralNetwork::adjustWeightsRPROP(float increase, float decrease, bool initialUpdate) {
    // Initial Update Must Be Gradient Descent To Calculate Initial Deltas
    adjustWeightsGradientDescent(1.0f, 0.0f);

    // Update Parameters
    for (int i = 0; i < layerCount; i++) {
        int nodeCount = layerNodes[i].size();
        int biasCount = layerBiases[i].size();

        // Update Weights Coming Out of Nodes
        for (int n = 0; n < nodeCount; n++) {
            int weightCount = layerNodes[i][n].outWeights.size();

            for (int w = 0; w < weightCount; w++) {
                float newDelta = layerNodes[i][n].value * layerNodes[i + 1][w].derivativeErrorValue;
                float previousDelta = layerNodes[i][n].previousDeltas[w];

                // Increase Delta if new Delta is in same direction as last, if not then decrease
                if (newDelta * previousDelta > 0.0f) {
                    newDelta = newDelta * increase;
                }
                else {
                    newDelta = newDelta * decrease;
                }

                // Update Parameters
                layerNodes[i][n].outWeights[w] += newDelta;
                layerNodes[i][n].previousDeltas[w] = newDelta;

            }
        }

        // Update Weights Coming Out of Biases      
        for (int b = 0; b < biasCount; b++) {
            int outWeightCount = layerBiases[i][b].outWeights.size();

            for (int w = 0; w < outWeightCount; w++) {
                float newDelta = layerNodes[i + 1][w].derivativeErrorValue;
                float previousDelta = layerBiases[i][b].previousDeltas[w];

                // Increase Delta if new Delta is in same direction as last, if not then decrease
                if (newDelta * previousDelta > 0.0f) {
                    newDelta = newDelta * increase;
                }
                else {
                    newDelta = newDelta * decrease;
                }

                // Update Parameters
                layerBiases[i][b].outWeights[w] += newDelta;
                layerBiases[i][b].previousDeltas[w] = newDelta;
            }
        }
    }
}

// Natural Selection measuring how good a network is
vector<float> NeuralNetwork::softmax(vector<float> input) {
    vector<float> result = input;

    // To avoid high computation, subtract largest value from all - produces same results
    int size = input.size();
    float maxValue = input[0];
    
    for (int i = 0; i < size; i++) {
        maxValue = fmaxf(abs(input[i]), maxValue);
    }
    for (int i = 0; i < size; i++) {
        
        result[i] = input[i] / maxValue;
    }
    
    // Peform softmax
    float total = 0.0f;
    for (int i = 0; i < size; i++) {
        total = total + expf(result[i]);
    }

    for (int i = 0; i < size; i++) {
        input[i] = expf(result[i]) / total;
    }
    
    return input;
}
float NeuralNetwork::measureNetworkFitness(NeuralNetwork network, standardTrainConfig trainConfig, vector<vector<float>> usedInputs, vector<vector<float>> usedOutputs) {
    // A Higher Fitness is better, meaning further negative is worse
    int datasetSize = usedInputs.size();
    int outputCount = usedOutputs[0].size();
    float currentFitness = 0.0f;

    for (int j = 0; j < datasetSize; j++) {
        vector<float> predicted = network.predict(usedInputs[j]);

        for (int k = 0; k < outputCount; k++) {
            // Update total error according to learning parameter chosen
            if (trainConfig.naturalSelection.fitnessFunctionType == ABSOLUTE_ERROR) {
                currentFitness = currentFitness - abs(usedOutputs[j][k] - predicted[k]);
            }
            if (trainConfig.naturalSelection.fitnessFunctionType == SQUARED_ERROR || trainConfig.naturalSelection.fitnessFunctionType == ROOT_SQUARED_ERROR || trainConfig.naturalSelection.fitnessFunctionType == MEAN_SQUARED_ERROR) {
                currentFitness = currentFitness - powf(usedOutputs[j][k] - predicted[k], 2.0f);
            }
        }
    }

    // calculate final fitness according to learnign parameter chosen
    if (trainConfig.naturalSelection.fitnessFunctionType == ROOT_SQUARED_ERROR) {
        currentFitness = sqrtf(currentFitness);
    }
    if (trainConfig.naturalSelection.fitnessFunctionType == MEAN_SQUARED_ERROR) {
        currentFitness = (1.0f / float(datasetSize)) * currentFitness;
    }
    if (trainConfig.naturalSelection.useStochasticDataset) {
        currentFitness = currentFitness * (float(trainConfig.trainInputs.size()) / float(trainConfig.naturalSelection.stochasticDatasetSize));
    }

    return currentFitness;
}
vector<float> NeuralNetwork::measurePopulationFitness(vector<NeuralNetwork> population, standardTrainConfig trainConfig) {
    int populationSize = population.size();

    vector<vector<float>> usedInputs = trainConfig.trainInputs;
    vector<vector<float>> usedOutputs = trainConfig.trainOutputs;

    // create a mini dataset if stochastic training is used
    if (trainConfig.naturalSelection.useStochasticDataset) {
        int miniBatchSize = trainConfig.trainInputs.size() / trainConfig.naturalSelection.stochasticDatasetSize;

        usedInputs.clear();
        usedOutputs.clear();

        // take random samples at regular intervals
        for (int i = 0; i < trainConfig.naturalSelection.stochasticDatasetSize; i++) {
            int newIndex = (i * miniBatchSize) + (rand() % miniBatchSize);

            usedInputs.push_back(trainConfig.trainInputs[newIndex]);
            usedOutputs.push_back(trainConfig.trainOutputs[newIndex]);
        }
    }
    
    vector<float> result;
    int populationCount = population.size();

    if (!trainConfig.naturalSelection.useThreading) {
        // run networks one after another

        for (int i = 0; i < populationCount; i++) {
            float currentFitness = measureNetworkFitness(population[i], trainConfig, usedInputs, usedOutputs);
            result.push_back(currentFitness);
        }
    }
    if (trainConfig.naturalSelection.useThreading) {
        vector<shared_future<float>> threads;

        // Run all networks simultanously
        for (int i = 0; i < populationCount; i++) {
            shared_future<float> future = async(measureNetworkFitness, population[i], trainConfig, usedInputs, usedOutputs);
            threads.push_back(future);
        }
        for (int i = 0; i < populationCount; i++) {
            float returnedFitnessValue = threads[i].get();
            result.push_back(returnedFitnessValue);
        }
    }

    return result;
}

// sorting and reproducing networks according to how good they were measured
NeuralNetwork NeuralNetwork::reproduceParents(vector<NeuralNetwork> parents, vector<float> fitnessScores, standardTrainConfig trainConfig) {
    NeuralNetwork result = parents[0];
    if (trainConfig.naturalSelection.breedingMethod == AVERAGE_PARENTS) {
        int parentCount = parents.size();

        // Add up all weights and take mean
        for (int i = 1; i < parentCount; i++) {
            int layerCount = result.layerNodes.size();

            for (int l = 0; l < layerCount - 1; l++) {
                int nodeCount = result.layerNodes[l].size();
                int biasCount = result.layerBiases[l].size();
                int weightCount = result.layerNodes[l + 1].size();

                for (int n = 0; n < nodeCount; n++) {
                    for (int w = 0; w < weightCount; w++) {
                        result.layerNodes[l][n].outWeights[w] = result.layerNodes[l][n].outWeights[w] + parents[i].layerNodes[l][n].outWeights[w] * (1.0f / parentCount);
                    }
                }
                for (int b = 0; b < biasCount; b++) {
                    for (int w = 0; w < weightCount; w++) {
                        result.layerBiases[l][b].outWeights[w] = result.layerBiases[l][b].outWeights[w] + parents[i].layerBiases[l][b].outWeights[w] * (1.0f / parentCount);
                    }
                }
            }
        }
    }
    if (trainConfig.naturalSelection.breedingMethod == WEIGHTED_PARENTS) {
        // Apply softmax to fitness scores to make them multiplicable to weights
        fitnessScores = softmax(fitnessScores);
        
        // Create output
        int populationSize = parents.size();

        for (int i = 0; i < populationSize; i++) {
            int layerCount = result.layerNodes.size();

            for (int l = 0; l < layerCount - 1; l++) {
                int nodeCount = result.layerNodes[l].size();
                int biasCount = result.layerBiases[l].size();
                int weightCount = result.layerNodes[l + 1].size();

                for (int n = 0; n < nodeCount; n++) {
                    for (int w = 0; w < weightCount; w++) {
                        float newWeight = result.layerNodes[l][n].outWeights[w];
                        if (i == 0) {
                            newWeight = 0.0f;
                        }
                        
                        newWeight = newWeight + parents[i].layerNodes[l][n].outWeights[w] * fitnessScores[i];
                        result.layerNodes[l][n].outWeights[w] = newWeight;
                    }
                }
                for (int b = 0; b < biasCount; b++) {
                    for (int w = 0; w < weightCount; w++) {
                        float newWeight = result.layerBiases[l][b].outWeights[w];
                        if (i == 0) {
                            newWeight = 0.0f;
                        }

                        newWeight = newWeight + parents[i].layerBiases[l][b].outWeights[w] * fitnessScores[i];
                        result.layerBiases[l][b].outWeights[w] = newWeight;
                    }
                }
            }
        }
    }

    if (trainConfig.naturalSelection.useChildMutation) {
        result = mutateNetwork(result);
    }
    return result;
}
vector<NeuralNetwork> NeuralNetwork::sortNetworks(vector<NeuralNetwork> networks, vector<float> fitnessScores) {
    // Bubble sort algorithm due to fitness scores
    int size = networks.size();

    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (fitnessScores[j] > fitnessScores[j + 1]) {
                float oldValue = fitnessScores[j];
                NeuralNetwork oldNetwork = networks[j];

                fitnessScores[j] = fitnessScores[j + 1];
                networks[j] = networks[j + 1];

                fitnessScores[j + 1] = oldValue;
                networks[j + 1] = oldNetwork;
            }
        }
    }

    return networks;
}
pair<NeuralNetwork, float> NeuralNetwork::chooseParent(vector<NeuralNetwork> population, vector<float> fitnessScores, standardTrainConfig trainConfig) {
    if (trainConfig.naturalSelection.parentSelectionMethod == TOP_PARENTS) {
        // Take random from top 10% of population
        int maxIndex = ceilf(float(population.size()) / 10.0f);
        int randomIndex = rand() % maxIndex;

        return make_pair(population[randomIndex], fitnessScores[randomIndex]);
    }
    if (trainConfig.naturalSelection.parentSelectionMethod == EXPONENTIAL_PARENTS) {
        // Use exponential distribution to choose index of sorted list
        int populationSize = population.size();
        
        float distributionParameter = fmaxf(1.0f, sqrtf(populationSize));
        exponential_distribution<float> distribution(distributionParameter);
        
        float distributionOutput = fminf(distribution(generator) * populationSize, populationSize - 1);
        int exponentialIndex = floorf(distributionOutput);
        
        return make_pair(population[exponentialIndex], fitnessScores[exponentialIndex]);
    }
    if (trainConfig.naturalSelection.parentSelectionMethod == PROBABILITY_PARENTS) {
        // Apply softmax to fitness scores
        fitnessScores = softmax(fitnessScores);

        // Find parent according to softmax fitness probabilities
        int populationSize = population.size();
        float randomProbability = double(rand()) / RAND_MAX; // creates probability in range 0 -> 1
        
        NeuralNetwork chosenParent = population[0];
        float chosenFitness = fitnessScores[0];

        float accumulativeTotal = 0.0f;
        for (int k = 0; k < populationSize; k++) {
            if (randomProbability < accumulativeTotal + fitnessScores[k]) {
                chosenParent = population[k];
                chosenFitness = fitnessScores[k];

                break;
            }
            accumulativeTotal = accumulativeTotal + fitnessScores[k];
        }

        return make_pair(chosenParent, chosenFitness);
    }
}

// reproducing a full population of architechtures according to specific rules
vector<NeuralNetwork> NeuralNetwork::reproducePopulation(vector<NeuralNetwork> parentPopulation, vector<float> fitnessScores, standardTrainConfig trainConfig) {
    int populationSize = parentPopulation.size();
    
    if (trainConfig.naturalSelection.parentSelectionMethod == TOP_PARENTS || trainConfig.naturalSelection.parentSelectionMethod == EXPONENTIAL_PARENTS) {
        parentPopulation = sortNetworks(parentPopulation, fitnessScores);
    }

    // Create Population
    vector<NeuralNetwork> result;
    vector<shared_future<NeuralNetwork>> threads;

    for (int i = 0; i < populationSize; i++) {
        // Find parents according to softmax fitness probabilities
        vector<NeuralNetwork> parents;
        vector<float> correspondingFitness;

        for (int j = 0; j < trainConfig.naturalSelection.parentCount; j++) {
            pair<NeuralNetwork, float> chosenParent = chooseParent(parentPopulation, fitnessScores, trainConfig);

            parents.push_back(chosenParent.first);
            correspondingFitness.push_back(chosenParent.second);
        }

        if (!trainConfig.naturalSelection.useThreading) {
            // Reproduce with parents
            NeuralNetwork child = reproduceParents(parents, correspondingFitness, trainConfig);
            result.push_back(child);
        }
        if (trainConfig.naturalSelection.useThreading) {
            shared_future<NeuralNetwork> future = async(reproduceParents, parents, correspondingFitness, trainConfig);
            threads.push_back(future);
        }
    }

    if (trainConfig.naturalSelection.useThreading) {
        for (int i = 0; i < populationSize; i++) {
            NeuralNetwork returnedChild = threads[i].get();
            result.push_back(returnedChild);
        }
    }

    return result;
}
NeuralNetwork NeuralNetwork::mutateNetwork(NeuralNetwork input) {
    // Add small random values (returned by normalDistribution()) to each weight, creating slightly more variation    
    int layerCount = input.layerCount;

    for (int l = 0; l < layerCount - 1; l++) {
        int nodeCount = input.layerNodes[l].size();
        int biasCount = input.layerBiases[l].size();
        int weightCount = input.layerNodes[l + 1].size();

        for (int n = 0; n < nodeCount; n++) {
            for (int w = 0; w < weightCount; w++) {
                input.layerNodes[l][n].outWeights[w] += normalDistribution();
            }
        }
        for (int b = 0; b < biasCount; b++) {
            for (int w = 0; w < weightCount; w++) {
                input.layerBiases[l][b].outWeights[w] += normalDistribution();
            }
        }
    }

    return input;
}

// initialising basic networks
vector<NeuralNetwork> NeuralNetwork::initialisePopulation(vector<int> layers, vector<int> biases, vector<int> activations, int count, float lowestWeight, float highestWeight) {
    randomMinimum = lowestWeight;
    randomMaximum = highestWeight;

    vector<NeuralNetwork> result;
    for (int i = 0; i < count; i++) {
        NeuralNetwork newNetwork = NeuralNetwork(layers, biases, activations);
        result.push_back(newNetwork);
    }

    randomMinimum = -1.0f;
    randomMaximum = 1.0f;

    return result;
}
NeuralNetwork NeuralNetwork::trainNaturalSelectionMethod(standardTrainConfig trainConfig, vector<int> layers, vector<int> biases, vector<int> activations) {
    // get a set of networks and get them ready 
    vector<NeuralNetwork> population = initialisePopulation(layers, biases, activations, trainConfig.naturalSelection.population, trainConfig.naturalSelection.lowestInitialisedWeight, trainConfig.naturalSelection.highestInitialisedWeight);
    cout << "Initialised population.. " << endl;

    // setup relevant node attributes and vectors
    for (int i = 0; i < trainConfig.naturalSelection.population; i++) {
        population[i].setupNetworkForTraining(trainConfig);
    }

    NeuralNetwork bestNetwork = population[0];
    float bestFitness = -1.0f;

    // training
    for (int i = 0; i < trainConfig.epochs; i++) {
        // finding best network, then breed them
        vector<float> currentFitnessScores = measurePopulationFitness(population, trainConfig);

        float lowestFitness = currentFitnessScores[0];
        int lowestFitnessIndex = 0;

        // find best network in current iteration
        for (int j = 1; j < trainConfig.naturalSelection.population; j++) {
            if (currentFitnessScores[j] > lowestFitness) {
                lowestFitness = currentFitnessScores[j];
                lowestFitnessIndex = j;
            }
        }

        // compare current best network to alltime best network
        if (lowestFitness > bestFitness || bestFitness == -1.0f) {
            bestFitness = lowestFitness;
            bestNetwork = population[lowestFitnessIndex];
        }

        cout << "Epoch: " << i + 1 << " / " << trainConfig.epochs << ", Fitness: " << -lowestFitness << endl;
        population = reproducePopulation(population, currentFitnessScores, trainConfig);
    }

    return bestNetwork;
}

// Random Weights Method
void NeuralNetwork::randomizeWeights() {
    for (int i = 0; i < layerCount - 1; i++) {
        int nodeCount = layerNodes[i].size();
        int outWeightCount = layerNodes[i + 1].size();

        for (int n = 0; n < nodeCount; n++) {
            for (int w = 0; w < outWeightCount; w++) {
                layerNodes[i][n].outWeights[w] = randomFloat();
            }
        }

        int biasCount = layerBiases[i].size();

        for (int b = 0; b < biasCount; b++) {
            for (int w = 0; w < outWeightCount; w++) {
                layerBiases[i][b].outWeights[w] = randomFloat();
            }
        }
    }
}
vector<float> NeuralNetwork::trainRandomMethod(standardTrainConfig trainConfig) {
    vector<float> result;

    // Useful Integers Calculated Before Iteration
    float minimumFoundError = numeric_limits<float>().max();
    int trainDataCount = trainConfig.trainInputs.size();
    int outputCount = trainConfig.trainOutputs[0].size();

    // Find Random Parameters and Find if It Meets Threshold
    for (int epoch = 0; epoch < trainConfig.epochs; epoch++) {
        randomizeWeights();

        // Score Network For Comparison To Others
        float accumulativeError = 0.0f;
        for (int t = 0; t < trainDataCount; t++) {
            vector<float> predicted = predict(trainConfig.trainInputs[t]);

            for (int o = 0; o < outputCount; o++) {
                accumulativeError += abs(trainConfig.trainOutputs[t][o] - predicted[o]);
            }
        }

        cout << "Random Epoch: " << epoch + 1 << " / " << trainConfig.epochs << ", Error: " << accumulativeError << endl;
        result.push_back(accumulativeError);

        // If Network if 'Good Enough' Then Keep It
        if (accumulativeError < trainConfig.randomMethod.errorThreshold) {
            break;
        }

        minimumFoundError = min(minimumFoundError, accumulativeError);
    }

    cout << "Minimum Error Found: " << minimumFoundError << endl;

    return result;
}

// Levenberg Marquardt matrices for training
void NeuralNetwork::addDeltasLM(vector<float> deltas) {
    // Deltas Are Ordered By Nodes Then Biases Then Layers
    int index = 0;

    for (int i = 0; i < layerCount - 1; i++) {
        // Prequisite Integers To Help Efficiency
        int nodeCount = layerNodes[i].size();
        int biasCount = layerBiases[i].size();
        int weightCount = layerNodes[i + 1].size();

        // Update Parameters For Nodes
        for (int j = 0; j < nodeCount; j++) {
            for (int k = 0; k < weightCount; k++) {
                layerNodes[i][j].outWeights[k] -= deltas[index];
                index += 1;
            }
        }

        // Update Parameters For Biases
        for (int j = 0; j < biasCount; j++) {
            for (int k = 0; k < weightCount; k++) {
                layerBiases[i][j].outWeights[k] -= deltas[index];
                index += 1;
            }
        }
    }
}
vector<float> NeuralNetwork::calculateDeltasLM(vector<vector<float>> jacobianMatrix, vector<vector<float>> costMatrix, float dampen) {
    int length = jacobianMatrix[0].size();

    // Calculate Deltas From Levenberg Marquardt Formula
    // Calculate Approximate Hessian
    Matrix transposedJacobian = transposeMatrix(jacobianMatrix);
    Matrix lengthIdentityMatrix = identityMatrix(length);
    lengthIdentityMatrix = scalarMultiply(dampen, lengthIdentityMatrix);

    Matrix squareJacobian = matrixMultiply(transposedJacobian, jacobianMatrix);
    squareJacobian = matrixAddition(squareJacobian, lengthIdentityMatrix);
    Matrix approximateHessian = inverseMatrix(squareJacobian, 0);

    // In the case that the matrix cannot be inverted
    if (approximateHessian.size() == 0) {
        return {};
    }

    // Calculate Deltas Using Hessian
    transposedJacobian = matrixMultiply(transposedJacobian, costMatrix);
    transposedJacobian = scalarMultiply(2.0f, transposedJacobian);
    Matrix weightDeltas = matrixMultiply(approximateHessian, transposedJacobian);

    // Return Deltas In One Vector
    vector<float> weightDeltasResult;
    for (int i = 0; i < length; i++) {
        float delta = weightDeltas[i][0];
        weightDeltasResult.push_back(delta);
    }

    return weightDeltasResult;
}
vector<float> NeuralNetwork::getJacobianRowLM() {
    vector<float> jacobianRow;

    // Order Derivative of Squared Cost With Respect To Each Weight
    for (int i = 0; i < layerCount - 1; i++) {
        // Prequisite Integers To Help Efficiency
        int nodeCount = layerNodes[i].size();
        int biasCount = layerBiases[i].size();
        int weightCount = layerNodes[i + 1].size();

        // Derivatives For Nodes
        for (int j = 0; j < nodeCount; j++) {
            for (int k = 0; k < weightCount; k++) {
                float costDerivative = layerNodes[i][j].value * layerNodes[i + 1][k].derivativeErrorValue;
                jacobianRow.push_back(costDerivative);
            }
        }

        // Derivatives For Biases
        for (int j = 0; j < biasCount; j++) {
            for (int k = 0; k < weightCount; k++) {
                float costDerivative = 1.0f * layerNodes[i + 1][k].derivativeErrorValue;
                jacobianRow.push_back(costDerivative);
            }
        }
    }

    return jacobianRow;
}

// training with matrices using levenberd marquardt algorithm
vector<float> NeuralNetwork::trainLevenbergMarquardt(standardTrainConfig trainConfig) {
    // Useful Integers for Saving Efficiency
    int trainDataCount = trainConfig.trainInputs.size();
    int outputCount = trainConfig.trainOutputs[0].size();

    // Useful for Randomizing Dataset
    vector<int> trainIndexes;
    for (int i = 0; i < trainDataCount; i++) {
        trainIndexes.push_back(i);
    }

    // Find Optimal Parameters
    vector<float> result;
    for (int epoch = 0; epoch < trainConfig.epochs; epoch++) {
        // Shuffle Dataset
        random_shuffle(trainIndexes.begin(), trainIndexes.end());
        if (trainConfig.useDropout) {
            randomlyDropNodes(trainConfig.nodeBiasDropoutProbability);
        }

        // Matrices Used For Calculating Deltas
        vector<vector<float>> fullJacobianMatrix;
        vector<vector<float>> currentCostOutputs;

        float currentTotalCost = 0.0f;
        float totalError = 0.0f;

        // Find Squared Errors
        for (int t = 0; t < trainDataCount; t++) {
            int currentIndex = trainIndexes[t];
            vector<float> result = predict(trainConfig.trainInputs[currentIndex]);

            float costOutput = 0.0f;
            vector<float> errors;
            
            for (int e = 0; e < outputCount; e++) {
                float error = trainConfig.trainOutputs[currentIndex][e] - result[e];

                totalError += abs(error);
                currentTotalCost += powf(error, 2.0f);
                costOutput += powf(error, 2.0f);
                errors.push_back(error);
            }

            // 2.0f due to cost function derivative of x^2
            calculateDerivatives(errors, -2.0f); 
            fullJacobianMatrix.push_back(getJacobianRowLM());            
            currentCostOutputs.push_back({ costOutput });
        }

        vector<float> deltas = calculateDeltasLM(fullJacobianMatrix, currentCostOutputs, trainConfig.levenbergMarquardt.dampingParameter);

        // In the case that the hessian could not be inverted
        if (deltas.size() == 0) {
            trainConfig.levenbergMarquardt.dampingParameter *= trainConfig.levenbergMarquardt.dampIncreaseMultiplierLM;
            continue;
        }

        addDeltasLM(deltas);

        // Find New Cost
        float newError = 0.0f;
        for (int t = 0; t < trainDataCount; t++) {
            int currentIndex = trainIndexes[t];
            vector<float> result = predict(trainConfig.trainInputs[currentIndex]);

            for (int e = 0; e < outputCount; e++) {
                newError += powf(trainConfig.trainOutputs[currentIndex][e] - result[e], 2.0f);
            }
        }

        // Update Damping Parameter Accordinly
        if (newError > currentTotalCost) {
            trainConfig.levenbergMarquardt.dampingParameter *= trainConfig.levenbergMarquardt.dampIncreaseMultiplierLM;

            int deltaCount = deltas.size();
            for (int i = 0; i < deltaCount; i++) {
                deltas[i] *= -1.0f;
            }

            addDeltasLM(deltas);
        }
        else {
            trainConfig.levenbergMarquardt.dampingParameter *= trainConfig.levenbergMarquardt.dampDecreaseMultiplierLM;
        }

        // Reset Configs
        if (trainConfig.useWeightDecay) {
            decayWeights(trainConfig.weightDecayMultiplier);
        }
        reactivateNodes();

        cout << "Epoch: " << epoch + 1 << " / " << trainConfig.epochs << ", Total error from epoch: " << totalError << ", Layers: " << layerCount << endl;
        result.push_back(totalError);
    }

    return result;
}

// Batch Gradient Descent
float runPredictionThreadingGradientDescent(NeuralNetwork* network, vector<float> inputs, vector<float> outputs) {
    vector<float> prediction = network->predict(inputs);
    float result = 0.0f;
    
    // Calculate Differences
    int outputCount = outputs.size();
    vector<float> errors;

    for (int e = 0; e < outputCount; e++) {
        result = result + abs(outputs[e] - prediction[e]);
        errors.push_back(outputs[e] - prediction[e]);
    }
    
    network->calculateDerivatives(errors);

    // Add Derivatives To "Previous Deltas" Field
    network->addDerivativesBatchGradientDescent();

    return result;
}
vector<float> NeuralNetwork::trainBatchGradientDescent(standardTrainConfig trainConfig) {
    vector<float> result;

    vector<vector<float>> trainInputs = trainConfig.trainInputs;
    vector<vector<float>> trainOutputs = trainConfig.trainOutputs;

    for (int epoch = 0; epoch < trainConfig.epochs; epoch++) {
        // Find dataset
        if (trainConfig.gradientDescent.useAllSongDataset) {
            if (epoch % trainConfig.gradientDescent.datasetRefreshInterval == 0) {
                int chunksPerSong = trainConfig.gradientDescent.batchSize / (trainConfig.gradientDescent.allSongDatasetEnd - trainConfig.gradientDescent.allSongDatasetStart); // / song count
                pair<vector<vector<float>>, vector<vector<float>>> allSongMiniDataset = generateAllSongDataSet(trainConfig.gradientDescent.datasetAudioConfig, chunksPerSong, trainConfig.gradientDescent.allSongDatasetStart, trainConfig.gradientDescent.allSongDatasetEnd);

                trainInputs = allSongMiniDataset.first;
                trainOutputs = allSongMiniDataset.second;

                cout << "Input: " << trainInputs[0].size() << ", Output: " << trainOutputs[0].size() << endl;
            }           
        }

        int trainDataCount = trainInputs.size();
        int outputCount = trainOutputs[0].size();

        // Calculate Current Learning Rate
        float currentLearningRate = trainConfig.learningRate;
        float currentMomentum = trainConfig.momentum;

        if (trainConfig.gradientDescent.learningRateType == CYCLICAL_LEARNING_RATE) {
            // Calculate Multiplier For Learning Parameters such That The Multiplier Peaks at Half Epochs
            double currentCoefficient = double(epoch + 1) / (double(trainConfig.epochs));
            float value = 1.0f - abs(2.0f * (currentCoefficient - 0.5f));

            currentLearningRate = value * trainConfig.learningRate;
            currentMomentum = value * trainConfig.momentum;
        }

        if (trainConfig.gradientDescent.learningRateType == DECREASING_LEARNING_RATE) {
            float multiplier = float(epoch + 1) / float(trainConfig.epochs);
            currentLearningRate = (1.0f - multiplier) * trainConfig.learningRate;
            currentMomentum = multiplier * trainConfig.learningRate;
        }

        // Randomly Disable Some Nodes to Prevent Overfitting
        if (trainConfig.useDropout) {
            randomlyDropNodes(trainConfig.nodeBiasDropoutProbability);
        }

        // Accumulate Weight Deltas (no mean just yet)
        vector<shared_future<float>> threads;
        float totalError = 0.0f;

        for (int t = 0; t < trainDataCount; t++) {
            if (!trainConfig.gradientDescent.useThreading) {
                if (t % 500 == 0) {
                    //cout << epoch + 1 << ":" << t + 1 << "/" << trainDataCount << endl;
                }

                float predictionError = runPredictionThreadingGradientDescent(this, trainInputs[t], trainOutputs[t]);
                totalError = totalError + predictionError;
            }
            if (trainConfig.gradientDescent.useThreading) {
                shared_future<float> future = async(runPredictionThreadingGradientDescent, this, trainInputs[t], trainOutputs[t]);
                threads.push_back(future);
            }
        }

        // Get threads
        if (trainConfig.gradientDescent.useThreading) {
            for (int t = 0; t < trainDataCount; t++) {
                if (t % 25 == 0) {
                    cout << epoch + 1 << ":" << t + 1 << "/" << trainDataCount << endl;
                }

                float foundError = threads[t].get();
                totalError = totalError + foundError;
            }
       }

        // Update Parameters
        averageDerivativesBatchGradientDescent(trainDataCount);
        if (trainConfig.gradientDescent.learningRateType == ADAM_LEARNING_RATE) {
            updateNetworkBatchGradientDescentADAM(trainConfig);
        }
        else {
            updateNetworkBatchGradientDescent(currentLearningRate, currentMomentum);
        }

        // Reset Network
        zeroPreviousDeltasBatchGradientDescent();
        reactivateNodes();

        cout << "Epoch: " << epoch + 1 << " / " << trainConfig.epochs << ", Total Error : " << totalError << ", Layers: " << layerCount << ", LR:" << currentLearningRate << endl;
        result.push_back(totalError);
    }
    return result;
}

// updating weights in batch gradient descent according to chosen method
void NeuralNetwork::updateNetworkBatchGradientDescent(float learningRate, float momentum) {
    int layerCount = layerNodes.size();

    // update parameters
    for (int l = 0; l < layerCount - 1; l++) {
        int nodeCount = layerNodes[l].size();
        int biasCount = layerBiases[l].size();
        int weightCount = layerNodes[l + 1].size();

        // loop over each node and calculate each weight delta
        for (int n = 0; n < nodeCount; n++) {
            if (!layerNodes[l][n].active) {
                continue;
            }

            for (int w = 0; w < weightCount; w++) {
                float newDelta = layerNodes[l][n].accumulativeDeltas[w] * learningRate;
                layerNodes[l][n].outWeights[w] += newDelta;
            
                layerNodes[l][n].outWeights[w] += layerNodes[l][n].previousDeltas[w] * momentum;
                layerNodes[l][n].previousDeltas[w] = newDelta;
            }
        }

        // loop over each bias and calculate each weight delta
        for (int b = 0; b < biasCount; b++) {
            if (!layerBiases[l][b].active) {
                continue;
            }

            for (int w = 0; w < weightCount; w++) {
                float newDelta = layerBiases[l][b].accumulativeDeltas[w] * learningRate;
                layerBiases[l][b].outWeights[w] += newDelta;

                layerBiases[l][b].outWeights[w] += layerBiases[l][b].previousDeltas[w] * momentum;
                layerBiases[l][b].previousDeltas[w] = newDelta;
            }
        }
    }
}
void NeuralNetwork::updateNetworkBatchGradientDescentADAM(standardTrainConfig trainConfig) {
    for (int i = 0; i < layerCount; i++) {
        int nodeCount = layerNodes[i].size();

        // Adjust Weights That Come From Nodes
        for (int n = 0; n < nodeCount; n++) {
            if (!layerNodes[i][n].active) {
                continue;
            }

            int weightCount = layerNodes[i][n].outWeights.size();
            for (int w = 0; w < weightCount; w++) {
                // Calculate Gradient of Error With Respect To Node
                float gradient = layerNodes[i][n].accumulativeDeltas[w];

                // ADAM Parameters#
                float newExponential = trainConfig.gradientDescent.betaOne * layerNodes[i][n].previousExponentials[w] + (1 - trainConfig.gradientDescent.betaOne) * gradient;
                float newSquaredExponential = trainConfig.gradientDescent.betaTwo * layerNodes[i][n].previousSquaredExponentials[w] + (1 - trainConfig.gradientDescent.betaTwo) * powf(gradient, 2.0f);

                float currentExponential = newExponential / (1 - trainConfig.gradientDescent.betaOne);
                float currentSquaredExponential = newSquaredExponential / (1 - trainConfig.gradientDescent.betaTwo);

                float learningRate = trainConfig.learningRate * (currentExponential / (sqrtf(currentSquaredExponential) + trainConfig.gradientDescent.epsillon));
                float delta = learningRate * gradient;

                layerNodes[i][n].outWeights[w] -= delta;

                // Update ADAM Parameters
                layerNodes[i][n].previousExponentials[w] = newExponential;
                layerNodes[i][n].previousSquaredExponentials[w] = newSquaredExponential;
            }
        }

        // Adjust Weights That Come From Biases
        int biasCount = layerBiases[i].size();
        for (int b = 0; b < biasCount; b++) {
            if (!layerBiases[i][b].active) {
                continue;
            }

            int outWeightCount = layerBiases[i][b].outWeights.size();
            for (int w = 0; w < outWeightCount; w++) {
                // Calculate Gradient of Error With Respect To Node
                float gradient = layerBiases[i][b].outWeights[w];

                // ADAM Parameters
                float newExponential = trainConfig.gradientDescent.betaOne * layerBiases[i][b].previousExponentials[w] + (1 - trainConfig.gradientDescent.betaOne) * gradient;
                float newSquaredExponential = trainConfig.gradientDescent.betaTwo * layerBiases[i][b].previousSquaredExponentials[w] + (1 - trainConfig.gradientDescent.betaTwo) * powf(gradient, 2.0f);

                float currentExponential = newExponential / (1 - trainConfig.gradientDescent.betaOne);
                float currentSquaredExponential = newSquaredExponential / (1 - trainConfig.gradientDescent.betaTwo);

                float learningRate = trainConfig.learningRate * (currentExponential / (sqrtf(currentSquaredExponential) + trainConfig.gradientDescent.epsillon));
                float delta = learningRate * gradient;

                layerBiases[i][b].outWeights[w] -= delta;

                // Update ADAM Parameters
                layerBiases[i][b].previousExponentials[w] = newExponential;
                layerBiases[i][b].previousSquaredExponentials[w] = newSquaredExponential;
            }
        }
    }
}

// changing derivatives according to errors in batch gradient descent training
void NeuralNetwork::addDerivativesBatchGradientDescent() {
    int layerCount = layerNodes.size();

    for (int l = 0; l < layerCount - 1; l++) {
        int nodeCount = layerNodes[l].size();
        int biasCount = layerBiases[l].size();
        int weightCount = layerNodes[l + 1].size();

        // accumulate backward derivatives for each node
        for (int n = 0; n < nodeCount; n++) {
            for (int w = 0; w < weightCount; w++) {
                layerNodes[l][n].accumulativeDeltas[w] += layerNodes[l][n].value * layerNodes[l + 1][w].derivativeErrorValue;
            }
        }

        // for each bias
        for (int b = 0; b < biasCount; b++) {
            for (int w = 0; w < weightCount; w++) {
                layerBiases[l][b].accumulativeDeltas[w] += 1.0f * layerNodes[l + 1][w].derivativeErrorValue;
            }
        }
    }
}
void NeuralNetwork::averageDerivativesBatchGradientDescent(int count) {
    int layerCount = layerNodes.size();

    for (int l = 0; l < layerCount - 1; l++) {
        int nodeCount = layerNodes[l].size();
        int biasCount = layerBiases[l].size();
        int weightCount = layerNodes[l + 1].size();

        // average derivatives to take mean weight change across entire dataset
        for (int n = 0; n < nodeCount; n++) {
            for (int w = 0; w < weightCount; w++) {
                layerNodes[l][n].accumulativeDeltas[w] = layerNodes[l][n].accumulativeDeltas[w] / float(count);
            }
        }

        // across bias
        for (int b = 0; b < biasCount; b++) {
            for (int w = 0; w < weightCount; w++) {
                layerBiases[l][b].accumulativeDeltas[w] = layerBiases[l][b].accumulativeDeltas[w] / float(count);
            }
        }
    }
}
void NeuralNetwork::zeroPreviousDeltasBatchGradientDescent() {
    int layerCount = layerNodes.size();

    for (int l = 0; l < layerCount - 1; l++) {
        int nodeCount = layerNodes[l].size();
        int biasCount = layerBiases[l].size();
        int weightCount = layerNodes[l + 1].size();

        for (int n = 0; n < nodeCount; n++) {
            for (int w = 0; w < weightCount; w++) {
                layerNodes[l][n].accumulativeDeltas[w] = 0.0f;
            }
        }
        for (int b = 0; b < biasCount; b++) {
            for (int w = 0; w < weightCount; w++) {
                layerBiases[l][b].accumulativeDeltas[w] = 0.0f;
            }
        }
    }
}

// Big Train Algorithm For Finding Best Architechture
float NeuralNetwork::testNetworkArchitechture(vector<vector<float>> trainInputs, vector<vector<float>> trainOutputs, int epochs, int batchSize, int layerSize, int layerCount, int biasCount, float lr, float momentum) {
    int inputSize = trainInputs[0].size();
    int outputSize = trainOutputs[0].size();

    // Create Network
    vector<int> layers = { inputSize };
    for (int i = 0; i < layerCount; i++) {
        layers.push_back(layerSize);
    }
    layers.push_back(outputSize);

    vector<int> biases;
    vector<int> activations;

    for (int i = 0; i < layerCount + 2; i++) {
        biases.push_back(biasCount);
        activations.push_back(SIGMOID);
    }

    NeuralNetwork newNetwork = NeuralNetwork(layers, biases, activations);

    // Create Train Config
    standardTrainConfig newTrainConfig = standardTrainConfig();
    newTrainConfig.epochs = epochs;

    newTrainConfig.gradientDescent.learningRateType = CYCLICAL_LEARNING_RATE;
    newTrainConfig.learningRate = lr;
    newTrainConfig.momentum = momentum;

    newTrainConfig.trainInputs = trainInputs;
    newTrainConfig.trainOutputs = trainOutputs;

    newTrainConfig.trainType = STOCHASTIC_GRADIENT_DESCENT;
    newTrainConfig.gradientDescent.batchSize = batchSize;

    // Find Errors
    vector<float> errors = newNetwork.train(newTrainConfig);
    return errors[epochs - 1];
}
void NeuralNetwork::findBestArchitechture(architechtureFindingConfig config) {
    int chosenLayerSize = 0;
    int chosenLayerCount = 0;
    int chosenBiasCount = 0;

    float chosenLearningRate = 0.0f;
    float chosenMomentum = 0.0f;

    // Find best layer size 
    float currentLowestError = -1.0f;
    for (int i = -config.layerSizeVariation; i < config.layerSizeVariation + 1; i++) {
        int currentLayerSize = config.layerSize + i * config.layerSizeInterval;
        float returnedError = testNetworkArchitechture(config.trainInputs, config.trainOutputs, config.epochs, config.batchSize, currentLayerSize, config.layerCount, config.biasCount, config.learningRate, config.momentum);

        if (returnedError < currentLowestError || currentLowestError == -1.0f) {
            currentLowestError = returnedError;
            chosenLayerSize = currentLayerSize;
        }
    }

    // Find best layer count
    currentLowestError = -1.0f;
    for (int i = -config.layerCountVariation; i < config.layerCountVariation + 1; i++) {
        int currentLayerCount = config.layerCount + i;
        float returnedError = testNetworkArchitechture(config.trainInputs, config.trainOutputs, config.epochs, config.batchSize, config.layerSize, currentLayerCount, config.biasCount, config.learningRate, config.momentum);

        if (returnedError < currentLowestError || currentLowestError == -1.0f) {
            currentLowestError = returnedError;
            chosenLayerCount = currentLayerCount;
        }
    }

    // Find best bias count
    currentLowestError = -1.0f;
    for (int i = -config.biasCountVariation; i < config.biasCountVariation + 1; i++) {
        int currentBiasCount = config.biasCount + i;
        float returnedError = testNetworkArchitechture(config.trainInputs, config.trainOutputs, config.epochs, config.batchSize, config.layerSize, config.biasCount, currentBiasCount, config.learningRate, config.momentum);

        if (returnedError < currentLowestError || currentLowestError == -1.0f) {
            currentLowestError = returnedError;
            chosenBiasCount = currentBiasCount;
        }
    }

    // Find best learning rate
    currentLowestError = -1.0f;
    for (int i = -config.learningRateVariation; i < config.learningRateVariation + 1; i++) {
        float currentLearningRate = config.learningRate + i * config.learningRateInterval;
        float returnedError = testNetworkArchitechture(config.trainInputs, config.trainOutputs, config.epochs, config.batchSize, config.layerSize, config.biasCount, config.biasCount, currentLearningRate, config.momentum);

        if (returnedError < currentLowestError || currentLowestError == -1.0f) {
            currentLowestError = returnedError;
            chosenLearningRate = currentLearningRate;
        }
    }

    // Find best momentum
    currentLowestError = -1.0f;
    for (int i = -config.momentumVariation; i < config.momentumVariation + 1; i++) {
        float currentMomentum = config.momentum + i * config.momentumInterval;
        float returnedError = testNetworkArchitechture(config.trainInputs, config.trainOutputs, config.epochs, config.batchSize, config.layerSize, config.biasCount, config.biasCount, config.learningRate, currentMomentum);

        if (returnedError < currentLowestError || currentLowestError == -1.0f) {
            currentLowestError = returnedError;
            chosenMomentum = currentMomentum;
        }
    }

    // output results
    cout << "Best layer size: " << chosenLayerSize << endl;
    cout << "Best layer count: " << chosenLayerCount << endl;
    cout << "Best bias count: " << chosenBiasCount << endl;
    cout << "Best LR: " << chosenLearningRate << endl;
    cout << "Best momentum: " << chosenMomentum << endl;
}

// Natural Selection Algorithm for Finding Best Architechture
NeuralNetwork NeuralNetwork::architechtureNaturalSelection(standardTrainConfig trainConfig) {
    // generate networks and ready them for trainig
    vector<NeuralNetwork> population = initialiseArchitechturePopulation(trainConfig);
    cout << "Population initialised" << endl;

    for (int i = 0; i < trainConfig.naturalSelection.population; i++) {
        population[i].setupNetworkForTraining(trainConfig);
        //population[i].outputNetworkArchitechture();
    }

    NeuralNetwork bestNetwork = population[0];
    float bestFitness = -1.0f;

    // training
    for (int i = 0; i < trainConfig.epochs; i++) {
        // Find best network from fitness score then reproduce
        vector<NeuralNetwork*> populationAddress;
        for (int j = 0; j < trainConfig.naturalSelection.population; j++) {
            populationAddress.push_back(&population[j]);
        }

        // quantatively score each network
        vector<float> fitnessScores = measureArchitechturePopulationFitness(populationAddress, trainConfig);

        float lowestFitness = fitnessScores[0];
        int lowestFitnessIndex = 0;

        // find best network of current iteration
        for (int j = 1; j < trainConfig.naturalSelection.population; j++) {
            if (fitnessScores[j] < lowestFitness) {
                lowestFitness = fitnessScores[j];
                lowestFitnessIndex = j;
            }
        }

        // check if current best network is better than alltime best network
        if (lowestFitness > bestFitness || bestFitness == -1.0f) {
            bestFitness = lowestFitness;
            bestNetwork = population[lowestFitnessIndex];
        }

        cout << "Epoch: " << i + 1 << " / " << trainConfig.epochs << ", Fitness: " << -lowestFitness << endl;
        cout << "Best Architechture So Far:" << endl;
        bestNetwork.outputNetworkArchitechture(); 

        population = reproduceArchitechtureNetworks(population, fitnessScores, trainConfig);
    }

    return bestNetwork;
}
vector<NeuralNetwork> NeuralNetwork::initialiseArchitechturePopulation(standardTrainConfig trainConfig) {
    vector<NeuralNetwork> result;

    int inputSize = trainConfig.trainInputs[0].size();
    int outputSize = trainConfig.trainOutputs[0].size();

    for (int i = 0; i < trainConfig.naturalSelection.population; i++) {
        // Create config with random numbers within limits
        vector<int> layers = { inputSize };

        int firstLayerBiasCount = trainConfig.naturalSelection.selectionMinBias + (rand() % static_cast<int>(trainConfig.naturalSelection.selectionMaxBias - trainConfig.naturalSelection.selectionMinBias + 1));
        vector<int> biases = { firstLayerBiasCount };

        int chosenLayerCount = trainConfig.naturalSelection.selectionMinLayers + (rand() % static_cast<int>(trainConfig.naturalSelection.selectionMaxLayers - trainConfig.naturalSelection.selectionMinLayers + 1));
        for (int j = 1; j < chosenLayerCount - 2; j++) {
            int newLayerSize = trainConfig.naturalSelection.selectionMinNodes + (rand() % static_cast<int>(trainConfig.naturalSelection.selectionMaxNodes - trainConfig.naturalSelection.selectionMinNodes + 1));
            layers.push_back(newLayerSize);

            int newBiasCount = trainConfig.naturalSelection.selectionMinBias + (rand() % static_cast<int>(trainConfig.naturalSelection.selectionMaxBias - trainConfig.naturalSelection.selectionMinBias + 1));
            biases.push_back(newBiasCount);
        }

        layers.push_back(outputSize);
        biases.push_back(0);
        
        // find activations
        vector<int> activations = {};
        if (trainConfig.naturalSelection.selectionAllowedActivations == ACTIVATION_NONLINEAR_ONLY) {
            for (int j = 0; j < chosenLayerCount; j++) {
                int chosenActivation = 0 + (rand() % static_cast<int>(1 - 0 + 1)); // Sigmoid or TANH
                activations.push_back(chosenActivation);
            }
        }
        if (trainConfig.naturalSelection.selectionAllowedActivations == ACTIVATION_ALL) {
            for (int j = 0; j < chosenLayerCount; j++) {
                int chosenActivation = 0 + (rand() % static_cast<int>(5 - 0 + 1)); // All Activations
                activations.push_back(chosenActivation);
            }
        }
        if (trainConfig.naturalSelection.selectionAllowedActivations == ACTIVATION_SIGMOID_ONLY) {
            for (int j = 0; j < chosenLayerCount; j++) {
                int chosenActivation = SIGMOID;
                activations.push_back(chosenActivation);
            }
        }

        // generate final network config
        NeuralNetwork newNetwork = NeuralNetwork(layers, biases, activations);
        result.push_back(newNetwork);
    }
    return result;
}

// methods to quantatively measure how good a network architechture is
vector<float> NeuralNetwork::measureArchitechturePopulationFitness(vector<NeuralNetwork*> population, standardTrainConfig trainConfig) {
    vector<float> result;
    int populationCount = population.size();

    if (!trainConfig.naturalSelection.useThreading) {
        // calculate fitness of each network one after another
        for (int i = 0; i < populationCount; i++) {
            float currentFitness = population[i]->measureArchitechtureFitness(trainConfig);
            result.push_back(currentFitness);
        }
    }
    if (trainConfig.naturalSelection.useThreading) {
        vector<shared_future<float>> threads;

        // calculate fitness of each network simultaneously
        for (int i = 0; i < populationCount; i++) {
            shared_future<float> future = async(&NeuralNetwork::measureArchitechtureFitness, population[i], trainConfig);
            threads.push_back(future);
        }
        for (int i = 0; i < populationCount; i++) {
            float returnedFitnessValue = threads[i].get();
            result.push_back(returnedFitnessValue);
        }
    }

    return result;
}
float NeuralNetwork::measureArchitechtureFitness(standardTrainConfig trainConfig) {
    vector<vector<float>> usedInputs = trainConfig.trainInputs;
    vector<vector<float>> usedOutputs = trainConfig.trainOutputs;

    int errorMultiplier = 1;
    if (trainConfig.naturalSelection.useStochasticDataset) {
        int miniBatchSize = trainConfig.trainInputs.size() / trainConfig.naturalSelection.stochasticDatasetSize;
        errorMultiplier = miniBatchSize;

        usedInputs.clear();
        usedOutputs.clear();

        // Make stochastic dataset
        for (int i = 0; i < trainConfig.naturalSelection.stochasticDatasetSize; i++) {
            int newIndex = (i * miniBatchSize) + (rand() % miniBatchSize);

            usedInputs.push_back(trainConfig.trainInputs[newIndex]);
            usedOutputs.push_back(trainConfig.trainOutputs[newIndex]);
        }
    }

    int trainDataCount = usedInputs.size();
    int outputCount = usedOutputs[0].size();

    vector<int> trainIndexes;
    for (int i = 0; i < trainDataCount; i++) {
        trainIndexes.push_back(i);
    }

    float previousError = -1.0f;
    int counter = 0;

    while (true) {
        // Randomly Shuffle Dataset Indexes to Prevent Overfitting
        random_shuffle(trainIndexes.begin(), trainIndexes.end());

        // Calculate Current Learning Rate
        float currentLearningRate = trainConfig.learningRate;
        float currentMomentum = trainConfig.momentum;

        float totalError = 0.0f;
        for (int t = 0; t < trainDataCount; t++) {
            int currentIndex = trainIndexes[t];
            vector<float> result = predict(usedInputs[currentIndex]);

            // Calculate Differences In Actual Output
            vector<float> errors;
            for (int e = 0; e < outputCount; e++) {                
                totalError += abs(usedOutputs[currentIndex][e] - result[e]);
                errors.push_back(usedOutputs[currentIndex][e] - result[e]);
            }

            calculateDerivatives(errors);
            adjustWeightsGradientDescent(currentLearningRate, trainConfig.momentum);
        }
        
        // Check for convergence with chosen method
        if (abs(totalError - previousError) < trainConfig.naturalSelection.selectionConvergenceValue) {
            counter = counter + 1;
        }
        else {
            counter = 0;
        }
        if (counter == trainConfig.naturalSelection.selectionConvergenceCounter) { // Same Error Required 10 Times To Break
            break;
        }

        previousError = totalError;
    }

    cout << "Network finished, final error: " << previousError << endl;
    return previousError;
}

// reproduction of  networks by taking means of layer counts and node counts
vector<NeuralNetwork> NeuralNetwork::reproduceArchitechtureNetworks(vector<NeuralNetwork> population, vector<float> scores, standardTrainConfig trainConfig) {
    int populationSize = population.size();

    if (trainConfig.naturalSelection.parentSelectionMethod == TOP_PARENTS || trainConfig.naturalSelection.parentSelectionMethod == EXPONENTIAL_PARENTS) {
        population = sortNetworks(population, scores);
    }
    
    // Create Population
    vector<NeuralNetwork> result;
    vector<shared_future<NeuralNetwork>> threads;

    for (int i = 0; i < populationSize; i++) {
        // Find parents according to softmax fitness probabilities
        vector<NeuralNetwork> parents;
        vector<float> correspondingFitness;

        for (int j = 0; j < trainConfig.naturalSelection.parentCount; j++) {
            pair<NeuralNetwork, float> chosenParent = chooseParent(population, scores, trainConfig);

            parents.push_back(chosenParent.first);
            correspondingFitness.push_back(chosenParent.second);
        }

        if (!trainConfig.naturalSelection.useThreading) {
            // Reproduce with parents
            NeuralNetwork child = reproduceArchitechtureParents(parents, correspondingFitness, trainConfig);
            result.push_back(child);
        }
        if (trainConfig.naturalSelection.useThreading) {
            shared_future<NeuralNetwork> future = async(reproduceArchitechtureParents, parents, correspondingFitness, trainConfig);
            threads.push_back(future);
        }
    }

    if (trainConfig.naturalSelection.useThreading) {
        for (int i = 0; i < populationSize; i++) {
            NeuralNetwork returnedChild = threads[i].get();
            result.push_back(returnedChild);
        }
    }

    return result;
}
NeuralNetwork NeuralNetwork::reproduceArchitechtureParents(vector<NeuralNetwork> parents, vector<float> fitnessScores, standardTrainConfig trainConfig) {
    // Find current architechtures
    vector<pair<vector<int>, vector<int>>> parentArchitechtures;
    int parentCount = parents.size();

    vector<int> resultantLayers;
    vector<int> resultantBiases;

    int inputSize = trainConfig.trainInputs[0].size();
    int outputSize = trainConfig.trainOutputs[0].size();

    // Find amximum length of all architechtures to allow all to fit together
    int maximumLength = 0;
    for (int i = 0; i < parentCount; i++) {
        maximumLength = max(maximumLength, int(parents[i].layerNodes.size()));
    }
    maximumLength = maximumLength - 2; // Remove input and output layers
    
    for (int i = 0; i < parentCount; i++) {
        vector<int> currentLayers;
        vector<int> currentBiases = { (int) parents[i].layerBiases[0].size() };

        int layerCount = parents[i].layerNodes.size();
        for (int j = 1; j < layerCount - 1; j++) {
            currentLayers.push_back(parents[i].layerNodes[j].size());
            currentBiases.push_back(parents[i].layerBiases[j].size());
        }

        // Make all architechtures the same length in order to not mess with vector indexes
        int resultantSize = layerCount - 2;
        if (resultantSize < maximumLength) {
            int difference = maximumLength - resultantSize;

            for (int j = 0; j < difference; j++) {
                currentLayers.push_back(0);
                currentBiases.push_back(0);
            }
        }

        parentArchitechtures.push_back(make_pair(currentLayers, currentBiases));
    }

    // Average Parents Method
    if (trainConfig.naturalSelection.breedingMethod == AVERAGE_PARENTS) {
        // Take Mean Layer Count
        int layerAccumulation = 0;
        for (int i = 0; i < parentCount; i++) {
            layerAccumulation = layerAccumulation + parentArchitechtures[i].first.size();
        }
        int chosenLayerCount = int(layerAccumulation / parentCount);

        resultantLayers.push_back(inputSize);
        
        // Find first layer bias count
        int firstLayerBiasCount = 0;
        for (int j = 0; j < parentCount; j++) {
            firstLayerBiasCount = firstLayerBiasCount + parentArchitechtures[j].second[0];
        }
        resultantBiases.push_back(firstLayerBiasCount / parentCount);

        for (int i = 0; i < chosenLayerCount; i++) {
            int accumulativeLayerSize = 0;
            int accumulativeBiasSize = 0;

            // Find Average Layer & Bias Count in Layers
            for (int j = 0; j < parentCount; j++) {
                accumulativeLayerSize = accumulativeLayerSize + parentArchitechtures[j].first[i];
                accumulativeBiasSize = accumulativeBiasSize + parentArchitechtures[j].second[i + 1]; // +1 due to first layer bias count
            }

            resultantLayers.push_back(accumulativeLayerSize / parentCount);
            resultantBiases.push_back(accumulativeBiasSize / parentCount);
        }

        resultantLayers.push_back(outputSize);
        resultantBiases.push_back(0);
    }
    if (trainConfig.naturalSelection.breedingMethod == WEIGHTED_PARENTS) {
        vector<float> multipliers = softmax(fitnessScores);

        // Layer Count
        float accumulativeLayerCount = 0.0f;
        for (int i = 0; i < parentCount; i++) {
            accumulativeLayerCount = accumulativeLayerCount + float(parentArchitechtures[i].first.size()) * multipliers[i];
        }
        int chosenLayerCount = int(accumulativeLayerCount);

        resultantLayers.push_back(inputSize);

        // Find first layer bias count
        float firstLayerBiasCount = 0;
        for (int j = 0; j < parentCount; j++) {
            firstLayerBiasCount = firstLayerBiasCount + float(parentArchitechtures[j].second[0]) * multipliers[j];
        }
        resultantBiases.push_back(firstLayerBiasCount);

        for (int i = 0; i < chosenLayerCount; i++) {
            float accumulativeLayerSize = 0;
            float accumulativeBiasSize = 0;

            // Find Average Layer & Bias Count in Layers
            for (int j = 0; j < parentCount; j++) {
                accumulativeLayerSize = accumulativeLayerSize + float(parentArchitechtures[j].first[i]) * multipliers[j];
                accumulativeBiasSize = accumulativeBiasSize + float(parentArchitechtures[j].second[i + 1]) * multipliers[j]; // +1 due to first layer bias count
            }
            
            resultantLayers.push_back(accumulativeLayerSize);
            resultantBiases.push_back(accumulativeBiasSize);
        }

        resultantLayers.push_back(outputSize);
        resultantBiases.push_back(0);
    }

    // Mutate
    if (trainConfig.naturalSelection.useChildMutation) {
        pair<vector<int>, vector<int>> resultantMutation = mutateNetworkArchitechture(make_pair(resultantLayers, resultantBiases));
        resultantLayers = resultantMutation.first;
        resultantBiases = resultantMutation.second;
    }

    // Create Network
    vector<int> activations = {};
    int layerCount = resultantLayers.size();

    if (trainConfig.naturalSelection.selectionAllowedActivations == ACTIVATION_NONLINEAR_ONLY) {
        for (int j = 0; j < layerCount; j++) {
            int chosenActivation = 0 + (rand() % static_cast<int>(1 - 0 + 1)); // Sigmoid or TANH
            activations.push_back(chosenActivation);
        }
    }
    if (trainConfig.naturalSelection.selectionAllowedActivations == ACTIVATION_ALL) {
        for (int j = 0; j < layerCount; j++) {
            int chosenActivation = 0 + (rand() % static_cast<int>(5 - 0 + 1)); // All Activations
            activations.push_back(chosenActivation);
        }
    }
    if (trainConfig.naturalSelection.selectionAllowedActivations == ACTIVATION_SIGMOID_ONLY) {
        for (int j = 0; j < layerCount; j++) {
            int chosenActivation = SIGMOID;
            activations.push_back(chosenActivation);
        }
    }
    
    NeuralNetwork resultantNetwork = NeuralNetwork(resultantLayers, resultantBiases, activations);
    resultantNetwork.setupNetworkForTraining(trainConfig);
    
    return resultantNetwork;
}

// mutation and debugging
pair<vector<int>, vector<int>> NeuralNetwork::mutateNetworkArchitechture(pair<vector<int>, vector<int>> currentArchitechture) {
    // Make layers
    vector<int> layerResult = currentArchitechture.first;
    int layerCount = layerResult.size();

    for (int i = 1; i < layerCount - 1; i++) { // skip input & output layers
        int currentSize = layerResult[i];
        int maximumSizeDeviation = ceil(currentSize / 10); // Maximum 10% deviation

        int actualDeviation = ceil(architechtureNormalDistribution() * maximumSizeDeviation);
        int newLayerSize = currentSize + actualDeviation;

        if (newLayerSize <= 0) {
            newLayerSize = currentSize;
        }
        layerResult[i] = newLayerSize;
    }

    // Make biases
    vector<int> biasResult = currentArchitechture.second;
    for (int i = 1; i < layerCount - 1; i++) {
        int currentSize = biasResult[i];
        int maximumSizeDeviation = ceil(currentSize / 10); // Maximum 10% deviation

        int actualDeviation = ceil(architechtureNormalDistribution() * maximumSizeDeviation);
        int newLayerSize = currentSize + actualDeviation;

        if (newLayerSize < 0) {
            newLayerSize = currentSize;
        }
        biasResult[i] = newLayerSize;
    }

    // Return
    return make_pair(layerResult, biasResult);
}
void NeuralNetwork::outputNetworkArchitechture() {
    int layerCount = layerNodes.size();
    cout << "Output Network Architechture: " << endl;

    cout << "Layers: ";
    for (int i = 0; i < layerCount; i++) {
        cout << layerNodes[i].size() << ",";
    }
    cout << endl;

    cout << "Biases: ";
    for (int i = 0; i < layerCount; i++) {
        cout << layerBiases[i].size() << ",";
    }
    cout << endl;
}