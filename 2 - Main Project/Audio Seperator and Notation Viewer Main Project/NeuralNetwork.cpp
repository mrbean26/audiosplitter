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

    ifstream inputNodes;
    inputNodes.open(fileNameNodes, ios::in | ios::binary);
    inputNodes.read(reinterpret_cast<char*>(&allNodeWeights[0]), totalNodeWeightCount * sizeof(float));
    inputNodes.close();

    ifstream inputBiases;
    inputBiases.open(fileNameBias, ios::in | ios::binary);
    inputBiases.read(reinterpret_cast<char*>(&allBiasWeights[0]), totalBiasWeightCount * sizeof(float));
    inputBiases.close();

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
void NeuralNetwork::train(standardTrainConfig trainConfig) {
    if (trainConfig.trainType == STOCHASTIC_GRADIENT_DESCENT) {
        trainStochasticGradientDescent(trainConfig);
    }
    if (trainConfig.trainType == GRADIENT_DESCENT) {

    }
    if (trainConfig.trainType == RESISTANT_PROPAGATION) {
        trainResistantPropagation(trainConfig);
    }
    if (trainConfig.trainType == NATURAL_SELECTION) {
        trainNaturalSelectionMethod(trainConfig);
    }
    if (trainConfig.trainType == RANDOM_METHOD) {
        trainRandomMethod(trainConfig);
    }
    if (trainConfig.trainType == LEVENBERG_MARQUARDT) {
        trainLevenbergMarquardt(trainConfig);
    }
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
    resetDerivativesAndResults();

    int firstLayerCount = layerNodes[0].size();
    for (int i = 0; i < firstLayerCount; i++) {
        layerNodes[0][i].value = inputs[i];
    }

    for (int i = 0; i < layerCount; i++) {
        int thisLayerCount = layerNodes[i].size();

        if (i > 0) {
            for (int n = 0; n < thisLayerCount; n++) {
                layerNodes[i][n].value = activate(layerNodes[i][n].value);
            }
        }

        for (int n = 0; n < thisLayerCount; n++) {
            int outWeightCount = layerNodes[i][n].outWeights.size();

            if (!layerNodes[i][n].active) {
                continue;
            }

            for (int w = 0; w < outWeightCount; w++) {
                layerNodes[i + 1][w].value += layerNodes[i][n].value * layerNodes[i][n].outWeights[w];
            }
        }

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
    // work backwards
    for (int i = layerCount - 2; i > -1; i--) {
        int currentLayerCount = layerNodes[i].size();

        for (int n = 0; n < currentLayerCount; n++) {
            if (!layerNodes[i][n].active) {
                continue;
            }

            int outWeightCount = layerNodes[i][n].outWeights.size();

            float valueMultiplier = derivative(layerNodes[i][n].value);
            if (i == 0) {
                valueMultiplier = layerNodes[i][n].value;
            }

            for (int w = 0; w < outWeightCount; w++) {
                if (!layerNodes[i + 1][w].active) {
                    continue;
                }

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

// Gradient Descent
void NeuralNetwork::adjustWeightsGradientDescent(float lr, float momentum) {
    for (int i = 0; i < layerCount; i++) {
        int nodeCount = layerNodes[i].size();

        for (int n = 0; n < nodeCount; n++) {
            if (!layerNodes[i][n].active) {
                continue;
            }

            int weightCount = layerNodes[i][n].outWeights.size();

            for (int w = 0; w < weightCount; w++) {
                float newDelta = layerNodes[i][n].value * layerNodes[i + 1][w].derivativeErrorValue * lr;
                layerNodes[i][n].outWeights[w] += newDelta;

                layerNodes[i][n].outWeights[w] += layerNodes[i][n].previousDeltas[w] * momentum;
                layerNodes[i][n].previousDeltas[w] = newDelta;
            }
        }

        int biasCount = layerBiases[i].size();
        for (int b = 0; b < biasCount; b++) {
            if (!layerBiases[i][b].active) {
                continue;
            }

            int outWeightCount = layerBiases[i][b].outWeights.size();

            for (int w = 0; w < outWeightCount; w++) {
                float newDelta = 1.0f * layerNodes[i + 1][w].derivativeErrorValue * lr;
                layerBiases[i][b].outWeights[w] += newDelta;

                layerBiases[i][b].outWeights[w] += layerBiases[i][b].previousDeltas[w] * momentum;
                layerBiases[i][b].previousDeltas[w] = newDelta;
            }
        }
    }
}
vector<float> NeuralNetwork::trainStochasticGradientDescent(standardTrainConfig trainConfig) {
    int trainDataCount = trainConfig.trainInputs.size();
    int outputCount = trainConfig.trainOutputs[0].size();

    vector<int> trainIndexes;
    for (int i = 0; i < trainDataCount; i++) {
        trainIndexes.push_back(i);
    }

    vector<float> result;
    for (int epoch = 0; epoch < trainConfig.epochs; epoch++) {
        random_shuffle(trainIndexes.begin(), trainIndexes.end());

        float currentLearningRate = trainConfig.learningRate;
        float currentMomentum = trainConfig.momentum;

        if (trainConfig.useCyclicalLearningRateAndMomentum) { // Peak in Middle - use function hanning window
            double currentCoefficient = double(epoch + 1) / (double(trainConfig.epochs));

            /*
            * Hanning Function
            double pi = 3.14159265358979323846;
            double cosValue = cos(2 * pi * currentCoefficient);
            double value = 0.5 * (1 - cosValue);
            */

            //Linear Function
            float value = 1.0f - abs(2.0f * (currentCoefficient - 0.5f));

            currentLearningRate = value * trainConfig.learningRate;
            currentMomentum = (1 - value) * trainConfig.momentum;
        }

        if (trainConfig.useDropout) {
            randomlyDropNodes(trainConfig.nodeBiasDropoutProbability);
        }

        float totalError = 0.0f;
        for (int t = 0; t < trainDataCount; t++) {
            int currentIndex = trainIndexes[t];
            vector<float> result = predict(trainConfig.trainInputs[currentIndex]);

            vector<float> errors;

            float currentError = 0.0f;
            for (int e = 0; e < outputCount; e++) {
                currentError += abs(trainConfig.trainOutputs[currentIndex][e] - result[e]);
                errors.push_back(trainConfig.trainOutputs[currentIndex][e] - result[e]);
            }
            totalError += currentError;

            calculateDerivatives(errors);

            adjustWeightsGradientDescent(currentLearningRate, trainConfig.momentum);

            if (trainConfig.useWeightDecay) {
                decayWeights(trainConfig.weightDecayMultiplier);
            }
            //cout << "Epoch: " << epoch + 1 << " / " << epochs << ", Train data item: " << t + 1 << " / " << trainDataCount << ", Total Error: " << currentError << endl;
        }

        reactivateNodes();

        cout << "Epoch: " << epoch + 1 << " / " << trainConfig.epochs << ", Total error from epoch: " << totalError << ", Layers: " << layerCount << ", LR:" << currentLearningRate << endl;
        result.push_back(totalError);
    }

    return result;
}

// Resistant Propagation
vector<float> NeuralNetwork::trainResistantPropagation(standardTrainConfig trainConfig) {
    int trainDataCount = trainConfig.trainInputs.size();
    int outputCount = trainConfig.trainOutputs[0].size();

    vector<int> trainIndexes;
    for (int i = 0; i < trainDataCount; i++) {
        trainIndexes.push_back(i);
    }

    vector<float> result;
    for (int epoch = 0; epoch < trainConfig.epochs; epoch++) {
        random_shuffle(trainIndexes.begin(), trainIndexes.end());

        float totalError = 0.0f;
        for (int t = 0; t < trainDataCount; t++) {
            int currentIndex = trainIndexes[t];
            vector<float> result = predict(trainConfig.trainInputs[currentIndex]);

            vector<float> errors;

            float currentError = 0.0f;
            for (int e = 0; e < outputCount; e++) {
                currentError += abs(trainConfig.trainOutputs[currentIndex][e] - result[e]);
                errors.push_back(trainConfig.trainOutputs[currentIndex][e] - result[e]);
            }
            totalError += currentError;

            calculateDerivatives(errors);

            if (epoch == 0 && t == 0) {
                adjustWeightsGradientDescent(1.0f, 0.0f);
            }
            else {
                adjustWeightsRPROP(trainConfig.rpropWeightIncreaseMultiplier, trainConfig.rpropWeightDecreaseMultiplier);
            }

            if (trainConfig.useWeightDecay) {
                decayWeights(trainConfig.weightDecayMultiplier);
            }
        }

        cout << "Epoch: " << epoch + 1 << " / " << trainConfig.epochs << ", Total error from epoch: " << totalError << ", Layers: " << layerCount << endl;
        result.push_back(totalError);
    }

    return result;
}
void NeuralNetwork::adjustWeightsRPROP(float increase, float decrease) {
    for (int i = 0; i < layerCount; i++) {
        int nodeCount = layerNodes[i].size();

        for (int n = 0; n < nodeCount; n++) {
            int weightCount = layerNodes[i][n].outWeights.size();

            for (int w = 0; w < weightCount; w++) {
                float newDelta = layerNodes[i][n].value * layerNodes[i + 1][w].derivativeErrorValue;
                float previousDelta = layerNodes[i][n].previousDeltas[w];

                // Same Sign?
                if (newDelta * previousDelta > 0.0f) {
                    newDelta = newDelta * increase;
                }
                else {
                    newDelta = newDelta * decrease;
                }

                layerNodes[i][n].outWeights[w] += newDelta;
                layerNodes[i][n].previousDeltas[w] = newDelta;

            }
        }

        int biasCount = layerBiases[i].size();
        for (int b = 0; b < biasCount; b++) {
            int outWeightCount = layerBiases[i][b].outWeights.size();

            for (int w = 0; w < outWeightCount; w++) {
                float newDelta = layerNodes[i + 1][w].derivativeErrorValue;
                float previousDelta = layerBiases[i][b].previousDeltas[w];

                // Same Sign?
                if (newDelta * previousDelta > 0.0f) {
                    newDelta = newDelta * increase;
                }
                else {
                    newDelta = newDelta * decrease;
                }

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
    int trainCount = trainConfig.trainInputs.size();
    int trainOutputSize = trainConfig.trainOutputs[0].size();

    vector<float> result;
    for (int epoch = 0; epoch < trainConfig.epochs; epoch++) {
        float currentVariation = (float(trainConfig.epochs - (epoch + 1)) / float(trainConfig.epochs)) * trainConfig.initialVariation;
        float lowestErrorThisPopulation = numeric_limits<float>().max();

        vector<vector<Node>> bestNodesThisPopulation;
        vector<vector<Bias>> bestBiasesThisPopulation;

        for (int i = 0; i < trainConfig.population; i++) {
            layerNodes = randomNodeWeights(layerNodes, currentVariation);
            layerBiases = randomBiasWeights(layerBiases, currentVariation);

            float totalError = 0.0f;
            for (int t = 0; t < trainCount; t++) {
                vector<float> predicted = predict(trainConfig.trainInputs[t]);

                for (int o = 0; o < trainOutputSize; o++) {
                    totalError += abs(trainConfig.trainOutputs[t][o] - predicted[o]);
                }
            }

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
void NeuralNetwork::trainRandomMethod(standardTrainConfig trainConfig) {
    float minimumFoundError = 0.0f;

    int trainDataCount = trainConfig.trainInputs.size();
    int outputCount = trainConfig.trainOutputs[0].size();

    for (int epoch = 0; epoch < trainConfig.epochs; epoch++) {
        randomizeWeights();
        float accumulativeError = 0.0f;

        for (int t = 0; t < trainDataCount; t++) {
            vector<float> predicted = predict(trainConfig.trainInputs[t]);

            for (int o = 0; o < outputCount; o++) {
                accumulativeError = accumulativeError + abs(trainConfig.trainOutputs[t][o] - predicted[o]);

            }
        }

        cout << "Random Epoch: " << epoch + 1 << " / " << trainConfig.epochs << ", Error: " << accumulativeError << endl;

        if (accumulativeError < trainConfig.errorThreshold) {
            break;
        }

        minimumFoundError = min(minimumFoundError, accumulativeError);

        if (epoch == 0) {
            minimumFoundError = accumulativeError;
        }
    }

    cout << "Minimum Error Found: " << minimumFoundError << endl;
}

// Levenberg Marquardt
void NeuralNetwork::addDeltasLM(vector<float> deltas) {
    int index = 0;

    for (int i = 0; i < layerCount - 1; i++) {
        int nodeCount = layerNodes[i].size();
        int biasCount = layerBiases[i].size();
        int weightCount = layerNodes[i + 1].size();

        for (int j = 0; j < nodeCount; j++) {
            for (int k = 0; k < weightCount; k++) {
                layerNodes[i][j].outWeights[k] -= deltas[index];
                cout << deltas[index] << endl;
                index += 1;
            }
        }

        for (int j = 0; j < biasCount; j++) {
            for (int k = 0; k < weightCount; k++) {
                layerBiases[i][j].outWeights[k] -= deltas[index];
                cout << deltas[index] << endl;
                index += 1;
            }
        }
    }
}
vector<float> NeuralNetwork::calculateDeltasLM(float cost, float dampen) {
    // Get Jacobian in Order of layer nodes then layer biases
    vector<vector<float>> jacobianMatrix;
    vector<float> jacobianRow;

    for (int i = 0; i < layerCount - 1; i++) {
        int nodeCount = layerNodes[i].size();
        int biasCount = layerBiases[i].size();
        int weightCount = layerNodes[i + 1].size();

        for (int j = 0; j < nodeCount; j++) {
            for (int k = 0; k < weightCount; k++) {
                float costDerivative = layerNodes[i][j].value * layerNodes[i + 1][k].derivativeErrorValue;
                jacobianRow.push_back(costDerivative);
            }
        }

        for (int j = 0; j < biasCount; j++) {
            for (int k = 0; k < weightCount; k++) {
                float costDerivative = 1.0f * layerNodes[i + 1][k].derivativeErrorValue;
                jacobianRow.push_back(costDerivative);
            }
        }
    }

    jacobianMatrix.push_back(jacobianRow);
    int length = jacobianRow.size();

    // Calculate Deltas
    Matrix transposedJacobian = transposeMatrix(jacobianMatrix);
    Matrix lengthIdentityMatrix = identityMatrix(length);
    lengthIdentityMatrix = scalarMultiply(dampen, lengthIdentityMatrix);

    Matrix squareJacobian = matrixMultiply(transposedJacobian, jacobianMatrix);
    squareJacobian = matrixAddition(squareJacobian, lengthIdentityMatrix);
    Matrix approximateHessian = inverseMatrix(squareJacobian, 1);
    approximateHessian = matrixMultiply(approximateHessian, transposedJacobian);

    Matrix weightDeltas = scalarMultiply(cost, approximateHessian);
    vector<float> weightDeltasResult;

    for (int i = 0; i < length; i++) {
        float delta = weightDeltas[0][i];
        weightDeltasResult.push_back(delta);
    }


    return weightDeltasResult;
}
vector<float> NeuralNetwork::trainLevenbergMarquardt(standardTrainConfig trainConfig) {
    int trainDataCount = trainConfig.trainInputs.size();
    int outputCount = trainConfig.trainOutputs[0].size();

    vector<int> trainIndexes;
    for (int i = 0; i < trainDataCount; i++) {
        trainIndexes.push_back(i);
    }

    vector<float> result;
    for (int epoch = 0; epoch < trainConfig.epochs; epoch++) {
        random_shuffle(trainIndexes.begin(), trainIndexes.end());

        if (trainConfig.useDropout) {
            randomlyDropNodes(trainConfig.nodeBiasDropoutProbability);
        }

        float totalError = 0.0f;
        for (int t = 0; t < trainDataCount; t++) {
            int currentIndex = trainIndexes[t];
            vector<float> result = predict(trainConfig.trainInputs[currentIndex]);

            vector<float> errors;
            float currentError = 0.0f;
            float costOutput = 0.0f;

            for (int e = 0; e < outputCount; e++) {
                totalError = totalError + abs(trainConfig.trainOutputs[currentIndex][e] - result[e]);
                costOutput = costOutput + powf(trainConfig.trainOutputs[currentIndex][e] - result[e], 2.0f);
                errors.push_back(trainConfig.trainOutputs[currentIndex][e] - result[e]);
            }

            calculateDerivatives(errors, 2.0f); // 2.0f due to cost function derivative of x^2
            vector<float> weightDeltas = calculateDeltasLM(costOutput, trainConfig.dampingParameter);
            addDeltasLM(weightDeltas);

            // Approximate new cost here

            if (trainConfig.useWeightDecay) {
                decayWeights(trainConfig.weightDecayMultiplier);
            }
        }

        reactivateNodes();

        cout << "Epoch: " << epoch + 1 << " / " << trainConfig.epochs << ", Total error from epoch: " << totalError << ", Layers: " << layerCount << endl;
        result.push_back(totalError);
    }

    return result;
}