#include "Headers/NeuralNetwork.h"

#include <algorithm>
#include <functional> 
#include <numeric>

float randomFloat() {
    int number = rand() % 200;
    float result = (float)number / 100.0f - 1.0f;
    return result;
}

string vectorToString(vector<float> used) {
    string result = "(";

    int size = used.size();
    for (int i = 0; i < size; i++) {
        result += to_string(used[i]) + ", ";
    }

    result.pop_back();
    result.pop_back();

    return result + ")";
}

namespace activations {
    float sigmoid(float x) {
        return 1.0f / (1.0f + exp(-x));
    }

    float tanh(float x) {
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    }

    float relu(float x) {
        if (x > 0.0f) {
            return x;
        }
        return 0.0f;
    }

    float leaky_relu(float x) {
        if (x > 0.01f * x) {
            return x;
        }
        return x * 0.01f;
    }
}

namespace derivatives {
    float derivatives::sigmoid(float x) {
        return x * (1 - x);
    }

    float derivatives::tanh(float x) {
        return 1 - (x * x);
    }

    float derivatives::relu(float x) {
        if (x > 0.0f) {
            return 1.0f;
        }
        return 0.0f;
    }

    float derivatives::leaky_relu(float x) {
        if (x > 0.0f) {
            return 1.0f;
        }
        return 0.01f;
    }
}

const char * activationType = "sigmoid";
float activate(float x) {
    if (activationType == "sigmoid") {
        return activations::sigmoid(x);
    }
    if (activationType == "tanh") {
        return activations::tanh(x);
    }
    if (activationType == "relu") {
        return activations::relu(x);
    }
    if (activationType == "leaky_relu") {
        return activations::leaky_relu(x);
    }
    return activations::sigmoid(x);
}

float derive(float x) {
    if (activationType == "sigmoid") {
        return derivatives::sigmoid(x);
    }
    if (activationType == "tanh") {
        return derivatives::tanh(x);
    }
    if (activationType == "relu") {
        return derivatives::relu(x);
    }
    if (activationType == "leaky_relu") {
        return derivatives::leaky_relu(x);
    }
    return derivatives::sigmoid(x);
}

NeuralNetwork::NeuralNetwork(vector<int> layerNodeCount,
            vector<int> layerBiasCount, const char* activation) {
    
    activationType = activation;
    layerCount = layerNodeCount.size();

    nodeCounts = layerNodeCount;
    biasCounts = layerBiasCount;

    // initialise network
    int layerCount = layerNodeCount.size();
    for (int layerNum = 0; layerNum < layerCount; layerNum++) {
        vector<float> newValueVector(layerNodeCount[layerNum]);
        network.push_back(newValueVector);

        if (layerNum != layerCount - 1) {
            int size = layerNodeCount[layerNum] * layerNodeCount[layerNum + 1] + layerBiasCount[layerNum] * layerNodeCount[layerNum + 1];
            vector<float> newWeightVector(size);

            for (int w = 0; w < size; w++) {
                newWeightVector[w] = randomFloat();
            }
            network.push_back(newWeightVector);

            vector<float> newPreviousDeltas(size);
            previousDeltas.push_back(newPreviousDeltas);
        }
    }
}

vector<float> NeuralNetwork::feedForward(vector<float> inputs) {
    // add inputs to first layer of network
    resetValues();
    
    network[0].clear();
    network[0].insert(network[0].begin(), inputs.begin(), inputs.end());
    int networkSize = network.size();

    for (int layerNum = 0; layerNum < networkSize - 1; layerNum += 2) {
        int currentNodeCount = nodeCounts[layerNum / 2];
        int nextNodeCount = nodeCounts[(layerNum + 2) / 2];
        
        // node sums
        for (int currentLayerNodeNum = 0; currentLayerNodeNum < currentNodeCount; currentLayerNodeNum++) {
            int startWeightIndex = currentLayerNodeNum * nextNodeCount;
            vector<float> weights(network[layerNum + 1].begin() + startWeightIndex,
                                  network[layerNum + 1].begin() + startWeightIndex + nextNodeCount);

            vector<float> newAddition;
            transform(weights.begin(), weights.end(), back_inserter(newAddition),
                bind(multiplies<float>(), placeholders::_1, network[layerNum][currentLayerNodeNum]));

            
            transform(newAddition.begin(), newAddition.end(),
                network[layerNum + 2].begin(), network[layerNum + 2].begin(), plus<float>());
        }

        // bias sums
        for (int currentLayerBiasNum = 0; currentLayerBiasNum < biasCounts[layerNum / 2]; currentLayerBiasNum++) {
            int startWeightIndex = (currentNodeCount * nextNodeCount) + currentLayerBiasNum * nextNodeCount;
            vector<float> weights(network[layerNum + 1].begin() + startWeightIndex,
                network[layerNum + 1].begin() + startWeightIndex + nextNodeCount);

            transform(weights.begin(), weights.end(),
                network[layerNum + 2].begin(), network[layerNum + 2].begin(), plus<float>());
        }

        // activate
        transform(network[layerNum + 2].begin(), network[layerNum + 2].end(),
            network[layerNum + 2].begin(), activate);
    }

    return network[networkSize - 1];
}

void NeuralNetwork::train(int epochs, float learningRate, float momentum,
            vector<vector<float>> trainInputs, vector<vector<float>> trainOutputs) {
    int trainSize = trainInputs.size();
    int outputCount = trainOutputs[0].size();

    // for randomizing inputs so it doesnt learn next answer
    vector<int> trainIndexes;
    for (int i = 0; i < trainSize; i++) {
        trainIndexes.push_back(i);
    }

    // train
    for (int epoch = 0; epoch < epochs; epoch++) {
        random_shuffle(trainIndexes.begin(), trainIndexes.end());
        float percentageThrough = (float)(epoch + 1) / (float)(epochs);
        cout << percentageThrough * 100 << "%" << endl;

        for (int t = 0; t < trainSize; t++) {
            int currentIndex = trainIndexes[t];
            vector<float> currentPrediction = feedForward(trainInputs[currentIndex]);

            // subtract currentprediction from trainOutputs
            vector<float> errors(outputCount);
            transform(trainOutputs[currentIndex].begin(), trainOutputs[currentIndex].end(),
                currentPrediction.begin(), errors.begin(), minus<float>());

            // add derivatives & change weights
            setDerivativesAndDeltaWeights(errors, learningRate, momentum);
        }
    }
}

void NeuralNetwork::setDerivativesAndDeltaWeights(vector<float> errors, float learningRate, float momentum) {
    int networkLength = network.size();
    int finalNodeCount = network[networkLength - 1].size();
    
    transform(network[networkLength - 1].begin(), network[networkLength - 1].end(),
        network[networkLength - 1].begin(), derive);
    transform(network[networkLength - 1].begin(), network[networkLength - 1].end(),
        errors.begin(), network[networkLength - 1].begin(), multiplies<float>());

    for (int layerNumber = networkLength - 3; layerNumber >= 0; layerNumber -= 2) {
        int currentLayerNodeCount = nodeCounts[layerNumber / 2];
        int outWeightCount = nodeCounts[(layerNumber + 2) / 2];

        for (int node = 0; node < currentLayerNodeCount; node++) {
            float nodeValue = network[layerNumber][node];
            network[layerNumber][node] = 0.0f;

            float valueMultiplier = derive(nodeValue);
            if (layerNumber == 0) { valueMultiplier = nodeValue; }
            
            int weightStartIndex = node * outWeightCount;
            vector<float> outWeights(network[layerNumber + 1].begin() + weightStartIndex,
                network[layerNumber + 1].begin() + weightStartIndex + outWeightCount);
            
            vector<float> derivationAccumulative;
            transform(outWeights.begin(), outWeights.end(), back_inserter(derivationAccumulative), bind(multiplies<float>(), placeholders::_1, valueMultiplier));
            transform(derivationAccumulative.begin(), derivationAccumulative.end(), network[layerNumber + 2].begin(), derivationAccumulative.begin(), multiplies<float>());
            network[layerNumber][node] = accumulate(derivationAccumulative.begin(), derivationAccumulative.end(), 0.0f);


            vector<float> deltaWeights(network[layerNumber + 2].begin(), network[layerNumber + 2].end());
            transform(deltaWeights.begin(), deltaWeights.end(), deltaWeights.begin(), bind(multiplies<float>(), placeholders::_1, nodeValue));
            transform(deltaWeights.begin(), deltaWeights.end(), deltaWeights.begin(), bind(multiplies<float>(), placeholders::_1, learningRate));
            
            vector<float> momentumWeights;
            transform(previousDeltas[layerNumber / 2].begin(), previousDeltas[layerNumber / 2].end(), back_inserter(momentumWeights), bind(multiplies<float>(), placeholders::_1, momentum));
            transform(deltaWeights.begin(), deltaWeights.begin(), momentumWeights.begin(), deltaWeights.begin(), plus<float>());

            transform(deltaWeights.begin(), deltaWeights.end(), network[layerNumber + 1].begin() + weightStartIndex,
                network[layerNumber + 1].begin() + weightStartIndex, plus<float>());

            previousDeltas[layerNumber / 2] = deltaWeights;
        }

        int currentLayerBiasCount = biasCounts[layerNumber / 2];
        for (int bias = 0; bias < currentLayerBiasCount; bias++) {
            vector<float> deltaWeights(network[layerNumber + 2].begin(), network[layerNumber + 2].end());
            transform(deltaWeights.begin(), deltaWeights.end(), deltaWeights.begin(), bind(multiplies<float>(), placeholders::_1, learningRate));
            
            int weightStartIndex = currentLayerNodeCount * outWeightCount + bias * outWeightCount;
            transform(deltaWeights.begin(), deltaWeights.end(), network[layerNumber + 1].begin() + weightStartIndex,
                network[layerNumber + 1].begin() + weightStartIndex, plus<float>());
        }
    }
}

void NeuralNetwork::resetValues() {
    int networkLength = network.size();
    
    for (int layerNum = 0; layerNum < networkLength; layerNum += 2) {
        int nodeCount = nodeCounts[layerNum / 2];
        
        for (int node = 0; node < nodeCount; node++) {
            network[layerNum][node] = 0.0f;
        }
    }
}

void NeuralNetwork::runTests(vector<vector<float>> testInputs) {
    int count = testInputs.size();
    for (int i = 0; i < count; i++) {
        vector<float> output = feedForward(testInputs[i]);

        cout << "Running: " << vectorToString(testInputs[i]) << ", result : " << vectorToString(output) << endl;
    }
}