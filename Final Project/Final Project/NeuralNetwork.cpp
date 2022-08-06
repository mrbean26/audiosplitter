#include "Headers/NeuralNetwork.h"

#include <fstream>
#include <sstream>

#include <random>

#include <thread>
#include <future>

#include "Headers/files.h"

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

    if (trainConfig.trainType == GRADIENT_DESCENT) {
        result = trainGradientDescent(trainConfig);
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
        std::cout << vectorToString(predict(inputs[i])) << endl;
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

        std::cout << "Epoch: " << epoch + 1 << " / " << trainConfig.epochs << ", Total error from epoch: " << totalError << ", Layers: " << layerCount << ", LR:" << currentLearningRate << endl;
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