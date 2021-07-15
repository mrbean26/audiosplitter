#include "Headers/NeuralNetwork.h"

#include <fstream>
#include <sstream>

#include "Headers/files.h"
#include "Headers/matrices.h"

// Initialisation Functions
NeuralNetwork::NeuralNetwork(vector<int> layers, vector<int> biases, string activation) {
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
}
float randomFloat() {
    int number = rand() % 200;
    float result = (float)number / 100.0f - 1.0f;
    return result;
}
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

        for (int n = 0; n < currentNodeCount; n++) {
            for (int n1 = 0; n1 < nextNodeCount; n1++) {
                layerNodes[i][n].outWeights.push_back(randomFloat());
                layerNodes[i][n].previousDeltas.push_back(0.0f);
            }
        }

        int currentBiasCount = layerBiases[i].size();
        for (int b = 0; b < currentBiasCount; b++) {
            for (int n = 0; n < nextNodeCount; n++) {
                layerBiases[i][b].outWeights.push_back(randomFloat());
                layerBiases[i][b].previousDeltas.push_back(0.0f);
            }
        }
    }
}

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
vector<float> NeuralNetwork::train(standardTrainConfig trainConfig) {
    vector<float> result;

    if (trainConfig.trainType == STOCHASTIC_GRADIENT_DESCENT) {
        result = trainStochasticGradientDescent(trainConfig);
    }
    if (trainConfig.trainType == GRADIENT_DESCENT) {
        result = trainGradientDescent(trainConfig);
    }
    if (trainConfig.trainType == RESISTANT_PROPAGATION) {
        result = trainResistantPropagation(trainConfig);
    }
    if (trainConfig.trainType == NATURAL_SELECTION) {
        result = trainNaturalSelectionMethod(trainConfig);
    }
    if (trainConfig.trainType == RANDOM_METHOD) {
        result = trainRandomMethod(trainConfig);
    }
    if (trainConfig.trainType == LEVENBERG_MARQUARDT) {
        result = trainLevenbergMarquardt(trainConfig);
    }

    return result;
}

float NeuralNetwork::activate(float x) {
    if (activationType == "sigmoid") {
        return 1.0f / (1.0f + exp(-x));
    }
    if (activationType == "tanh") {
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    }
    if (activationType == "relu") {
        if (x > 0.0f) {
            return x;
        }
        return 0.0f;
    }
    if (activationType == "leaky_relu") {
        if (x > 0.01f * x) {
            return x;
        }
        return x * 0.01f;
    }
    return 1.0f / (1.0f + exp(-x));
}
float NeuralNetwork::derivative(float x) {
    if (activationType == "sigmoid") {
        return x * (1 - x);
    }
    if (activationType == "tanh") {
        return 1 - (x * x);
    }
    if (activationType == "relu") {
        if (x > 0.0f) {
            return 1.0f;
        }
        return 0.0f;
    }
    if (activationType == "leaky_relu") {
        if (x > 0.0f) {
            return 1.0f;
        }
        return 0.01f;
    }
    return x * (1 - x);
}

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

        // After Weighted Sum, Pass node value through activation function
        if (i > 0) {
            for (int n = 0; n < thisLayerCount; n++) {
                layerNodes[i][n].value = activate(layerNodes[i][n].value);
            }
        }

        // Add Weighted Sum To Nodes in Next Layers
        for (int n = 0; n < thisLayerCount; n++) {
            int outWeightCount = layerNodes[i][n].outWeights.size();

            if (!layerNodes[i][n].active) {
                continue;
            }

            for (int w = 0; w < outWeightCount; w++) {
                layerNodes[i + 1][w].value += layerNodes[i][n].value * layerNodes[i][n].outWeights[w];
            }
        }

        // Add Bias Weights (useful for when 0 values are present)
        int thisBiasCount = layerBiases[i].size();
        for (int b = 0; b < thisBiasCount; b++) {
            int outWeightCount = layerBiases[i][b].outWeights.size();

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

void NeuralNetwork::calculateDerivatives(vector<float> outputErrors, float errorMultiplier = 1.0f) {
    // with outputErrors as actual - target
    int finalLayerCount = layerNodes[layerCount - 1].size();

    for (int i = 0; i < finalLayerCount; i++) {
        layerNodes[layerCount - 1][i].derivativeErrorValue = derivative(layerNodes[layerCount - 1][i].value) * outputErrors[i] * errorMultiplier;
    }
    // Backpropagate by Calculating Partial Derivatives of Each Node with Respect to The Error
    for (int i = layerCount - 2; i > -1; i--) {
        int currentLayerCount = layerNodes[i].size();

        for (int n = 0; n < currentLayerCount; n++) {
            if (!layerNodes[i][n].active) {
                continue;
            }

            int outWeightCount = layerNodes[i][n].outWeights.size();
            float valueMultiplier = derivative(layerNodes[i][n].value);

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

void NeuralNetwork::decayWeights(float multiplier) {
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

// Dropout
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

        float currentLearningRate = trainConfig.learningRate;
        float currentMomentum = trainConfig.momentum;

        if (trainConfig.useCyclicalLearningRateAndMomentum) { 
            // Peak in Middle - Use Linear Function
            double currentCoefficient = double(epoch + 1) / (double(trainConfig.epochs));
            float value = 1.0f - abs(2.0f * (currentCoefficient - 0.5f));

            currentLearningRate = value * trainConfig.learningRate;
            currentMomentum = (1 - value) * trainConfig.momentum;
        }

        // Randomly Disable Some Nodes to Prevent Overfitting
        if (trainConfig.useDropout) {
            randomlyDropNodes(trainConfig.nodeBiasDropoutProbability);
        }

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

            calculateDerivatives(errors);
            adjustWeightsGradientDescent(currentLearningRate, trainConfig.momentum);

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

// Stochastic Gradient Descent (select a few random train inputs)
vector<float> NeuralNetwork::trainStochasticGradientDescent(standardTrainConfig trainConfig) {
    // Useful Integers Calculated Before Iteration
    int trainDataCount = trainConfig.trainInputs.size();
    int miniBatchSize = trainConfig.trainInputs.size() / trainConfig.batchSize;
    int outputCount = trainConfig.trainOutputs[0].size();

    // All Possible Indexes Across Dataset
    vector<int> trainIndexes;
    for (int i = 0; i < trainDataCount; i++) {
        trainIndexes.push_back(i);
    }

    vector<float> result;
    for (int epoch = 0; epoch < trainConfig.epochs; epoch++) {
        // Create Random Sectional Indexes for Stochastic Training Unless Epoch is Divisible by Full Dataset Test Parameter
        vector<int> trainIndexes;
        int currentBatchSize = 0;

        if ((epoch + 1) % trainConfig.entireBatchEpochIntervals == 0) {
            for (int i = 0; i < trainDataCount; i++) {
                trainIndexes.push_back(i);
            }

            currentBatchSize = trainDataCount;
        }
        else {
            for (int i = 0; i < trainConfig.batchSize; i++) {
                int newIndex = (i * miniBatchSize) + (rand() % miniBatchSize);
                trainIndexes.push_back(newIndex);
            }

            currentBatchSize = trainConfig.batchSize;
        }

        // Generate Learning Parameters
        float currentLearningRate = trainConfig.learningRate;
        float currentMomentum = trainConfig.momentum;

        if (trainConfig.useCyclicalLearningRateAndMomentum) { 
            // Calculate Multiplier For Learning Parameters such That The Multiplier Peaks at Half Epochs
            double currentCoefficient = double(epoch + 1) / (double(trainConfig.epochs));
            float value = 1.0f - abs(2.0f * (currentCoefficient - 0.5f));

            currentLearningRate = value * trainConfig.learningRate;
            currentMomentum = (1 - value) * trainConfig.momentum;
        }

        // Randomly Disable Some Nodes to Prevent Overfitting
        if (trainConfig.useDropout) {
            randomlyDropNodes(trainConfig.nodeBiasDropoutProbability);
        }

        // Adjust Parameters
        float totalError = 0.0f;
        for (int t = 0; t < trainIndexes.size(); t++) {
            int currentIndex = trainIndexes[t];
            vector<float> result = predict(trainConfig.trainInputs[currentIndex]);

            // Calculate Differences In Actual Output
            vector<float> errors;
            for (int e = 0; e < outputCount; e++) {
                totalError += abs(trainConfig.trainOutputs[currentIndex][e] - result[e]);
                errors.push_back(trainConfig.trainOutputs[currentIndex][e] - result[e]);
            }

            calculateDerivatives(errors);
            adjustWeightsGradientDescent(currentLearningRate, trainConfig.momentum);

            // Lower Some Weights To Prevent Overfitting
            if (trainConfig.useWeightDecay) {
                decayWeights(trainConfig.weightDecayMultiplier);
            }
        }

        // Reset Network
        reactivateNodes();

        float approximateTotalDatasetError = totalError * (float(trainDataCount) / float(currentBatchSize));
        cout << "Epoch: " << epoch + 1 << " / " << trainConfig.epochs << ", Approximate total error from epoch: " << approximateTotalDatasetError << ", Layers: " << layerCount << ", LR:" << currentLearningRate << endl;

        result.push_back(approximateTotalDatasetError);
    }

    return result;
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

            calculateDerivatives(errors);
            adjustWeightsRPROP(trainConfig.rpropWeightIncreaseMultiplier, trainConfig.rpropWeightDecreaseMultiplier, epoch == 0 && t == 0);

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

// Natural Selection
vector<vector<NeuralNetwork::Node>> NeuralNetwork::randomNodeWeights(vector<vector<Node>> initial, float variation) {
    int layerCount = initial.size();

    for (int l = 0; l < layerCount - 1; l++) {
        int nodeCount = initial[l].size();
        int weightCount = initial[l + 1].size();

        for (int n = 0; n < nodeCount; n++) {
            for (int w = 0; w < weightCount; w++) {
                initial[l][n].outWeights[w] += randomFloat() * variation;
            }
        }
    }

    return initial;
}
vector<vector<NeuralNetwork::Bias>> NeuralNetwork::randomBiasWeights(vector<vector<Bias>> initial, float variation) {
    int layerCount = initial.size();

    for (int l = 0; l < layerCount - 1; l++) {
        int nodeCount = initial[l].size();
        int weightCount = initial[l + 1].size();

        for (int n = 0; n < nodeCount; n++) {
            for (int w = 0; w < weightCount; w++) {
                initial[l][n].outWeights[w] += randomFloat() * variation;
            }
        }
    }

    return initial;
}
vector<float> NeuralNetwork::trainNaturalSelectionMethod(standardTrainConfig trainConfig) {
    // Useful Integers Calculated Before Iteration
    int trainCount = trainConfig.trainInputs.size();
    int trainOutputSize = trainConfig.trainOutputs[0].size();

    // Find Optimal Parameters
    vector<float> result;
    for (int epoch = 0; epoch < trainConfig.epochs; epoch++) {
        // Calculate Variation Accordingly With Epochs, Decreasing with Time
        float currentVariation = (float(trainConfig.epochs - (epoch + 1)) / float(trainConfig.epochs)) * trainConfig.initialVariation;
        float lowestErrorThisPopulation = numeric_limits<float>().max();

        // Used To Store Best Config
        vector<vector<Node>> bestNodesThisPopulation;
        vector<vector<Bias>> bestBiasesThisPopulation;

        // Initialise Several Random Configurations To Find Best In Population
        for (int i = 0; i < trainConfig.population; i++) {
            // Get Random Config
            layerNodes = randomNodeWeights(layerNodes, currentVariation);
            layerBiases = randomBiasWeights(layerBiases, currentVariation);

            // Score Network Config for Comparison Against Others
            float totalError = 0.0f;
            for (int t = 0; t < trainCount; t++) {
                vector<float> predicted = predict(trainConfig.trainInputs[t]);

                for (int o = 0; o < trainOutputSize; o++) {
                    totalError += abs(trainConfig.trainOutputs[t][o] - predicted[o]);
                }
            }

            // Compare and Update Current Configuration
            if (totalError < lowestErrorThisPopulation) {
                lowestErrorThisPopulation = totalError;
                bestNodesThisPopulation = layerNodes;
                bestBiasesThisPopulation = layerBiases;
            }
        }

        cout << "Epoch: " << epoch + 1 << " / " << trainConfig.epochs << ", Total Error: " << lowestErrorThisPopulation << endl;
        result.push_back(lowestErrorThisPopulation);

        layerNodes = bestNodesThisPopulation;
        layerBiases = bestBiasesThisPopulation;
    }

    return result;
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
        if (accumulativeError < trainConfig.errorThreshold) {
            break;
        }

        minimumFoundError = min(minimumFoundError, accumulativeError);
    }

    cout << "Minimum Error Found: " << minimumFoundError << endl;

    return result;
}

// Levenberg Marquardt
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

        vector<float> deltas = calculateDeltasLM(fullJacobianMatrix, currentCostOutputs, trainConfig.dampingParameter);

        // In the case that the hessian could not be inverted
        if (deltas.size() == 0) {
            trainConfig.dampingParameter *= trainConfig.dampIncreaseMultiplierLM;
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
            trainConfig.dampingParameter *= trainConfig.dampIncreaseMultiplierLM;

            int deltaCount = deltas.size();
            for (int i = 0; i < deltaCount; i++) {
                deltas[i] *= -1.0f;
            }

            addDeltasLM(deltas);
        }
        else {
            trainConfig.dampingParameter *= trainConfig.dampDecreaseMultiplierLM;
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