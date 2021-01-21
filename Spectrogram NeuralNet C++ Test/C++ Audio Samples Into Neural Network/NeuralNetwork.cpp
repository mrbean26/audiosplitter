#include "Headers/NeuralNetwork.h"

// activations
namespace activations{
    float sigmoid(float x){
        return 1.0f / (1.0f + exp(-x));
    }

    float tanh(float x){
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    }

    float relu(float x){
        if(x > 0.0f){
            return x;
        }
        return 0.0f;
    }

    float leaky_relu(float x){
        if(x > 0.01f * x){
            return x;
        }
        return x * 0.01f;
    }

    // derivatives
    float derivatives::sigmoid(float x){
        return x * (1 - x);
    }

    float derivatives::tanh(float x){
        return 1 - (x * x);
    }

    float derivatives::relu(float x){
        if(x > 0.0f){
            return 1.0f;
        }
        return 0.0f;
    }

    float derivatives::leaky_relu(float x){
        if(x > 0.0f){
            return 1.0f;
        }
        return 0.01f;
    }
}

// functions
float NeuralNetwork::activate(float x){
    if(activationType == "sigmoid"){
        return activations::sigmoid(x);
    }
    if(activationType == "tanh"){
        return activations::tanh(x);
    }
    if(activationType == "relu"){
        return activations::relu(x);
    }
    if(activationType == "leaky_relu"){
        return activations::leaky_relu(x);
    }
    return activations::sigmoid(x);
}

float NeuralNetwork::derivative(float x){
    if(activationType == "sigmoid"){
        return activations::derivatives::sigmoid(x);
    }
    if(activationType == "tanh"){
        return activations::derivatives::tanh(x);
    }
    if(activationType == "relu"){
        return activations::derivatives::relu(x);
    }
    if(activationType == "leaky_relu"){
        return activations::derivatives::leaky_relu(x);
    }
    return activations::derivatives::sigmoid(x);
}

float randomFloat(){
    int number = rand() % 200;
    float result = (float) number / 100.0f - 1.0f;
    return result;
}

string vectorToString(vector<float> used){
    string result = "(";

    int size = used.size();
    for(int i = 0; i < size; i++){
        result += to_string(used[i]) + ", ";
    }

    result.pop_back();
    result.pop_back();

    return result + ")";
}

void writeToFile(const char * fileName, vector<string> lines){
    ofstream currentFile;
	currentFile.open(fileName);

	if (!currentFile) {
		cout << "File could not be opened: " << fileName << endl;
		return;
	}

	int vectorSize = lines.size();
	for (int i = 0; i < vectorSize; i++) {
		currentFile << lines[i] << endl;
	}
	currentFile.close();
}

vector<string> readFile(const char * fileName){
    vector<string> result;

    ifstream newFile(fileName);
	string currentLine;

	if (!newFile) {
		cout << "File could not be opened: " << fileName << endl;
	}

	while (getline(newFile, currentLine)) {
		result.push_back(currentLine);
	}

    return result;
}

vector<string> splitStringByCharacter(string used, char splitter){
    vector<string> result;
    stringstream stringStream(used);

    while(stringStream.good()){
        string substring;
        getline(stringStream, substring, splitter);
        result.push_back(substring);
    }


    return result;
}

// Network
NeuralNetwork::NeuralNetwork(vector<int> layers, vector<int> biases, string activation){
    layerCount = layers.size();
    for(int i = 0; i < layerCount; i++){
        vector<Node> newLayer;
        for(int n = 0; n < layers[i]; n++){
            newLayer.push_back(Node());
        }
        layerNodes.push_back(newLayer);

        vector<Bias> newBiasLayer;
        for(int b = 0; b < biases[i]; b++){
            newBiasLayer.push_back(Bias());
        }
        layerBiases.push_back(newBiasLayer);
    }
    initialiseWeights();
}

void NeuralNetwork::initialiseWeights(){
    // clear weights
    for(int i = 0; i < layerCount; i++){
        int currentNodeCount = layerNodes[i].size();
        for(int n = 0; n < currentNodeCount; n++){
            layerNodes[i][n].outWeights.clear();
        }

        int currentBiasCount = layerBiases[i].size();
        for(int b = 0; b < currentBiasCount; b++){
            layerBiases[i][b].outWeights.clear();
        }
    }

    // initialise
    for(int i = 0; i < layerCount - 1; i++){
        int currentNodeCount = layerNodes[i].size();
        int nextNodeCount = layerNodes[i + 1].size();

        for(int n = 0; n < currentNodeCount; n++){
            for(int n1 = 0; n1 < nextNodeCount; n1++){
                layerNodes[i][n].outWeights.push_back(randomFloat());
                layerNodes[i][n].previousDeltas.push_back(0.0f);
            }
        }

        int currentBiasCount = layerBiases[i].size();
        for(int b = 0; b < currentBiasCount; b++){
            for(int n = 0; n < nextNodeCount; n++){
                layerBiases[i][b].outWeights.push_back(randomFloat());
                layerBiases[i][b].previousDeltas.push_back(0.0f);
            }
        }
    }
}

void NeuralNetwork::feedForward(vector<float> inputs){
    resetDerivativesAndResults();

    int firstLayerCount = layerNodes[0].size();
    for(int i = 0; i < firstLayerCount; i++){
        layerNodes[0][i].value = inputs[i];
    }

    for(int i = 0; i < layerCount; i++){
        int thisLayerCount = layerNodes[i].size();

        if(i > 0){
            for(int n = 0; n < thisLayerCount; n++){
                layerNodes[i][n].value = activate(layerNodes[i][n].value);
            }
        }

        for(int n = 0; n < thisLayerCount; n++){
            int outWeightCount = layerNodes[i][n].outWeights.size();

            for(int w = 0; w < outWeightCount; w++){
                layerNodes[i + 1][w].value += layerNodes[i][n].value * layerNodes[i][n].outWeights[w];
            }
        }

        int thisBiasCount = layerBiases[i].size();
        for(int b = 0; b < thisBiasCount; b++){
            int outWeightCount = layerBiases[i][b].outWeights.size();

            for(int w = 0; w < outWeightCount; w++){
                layerNodes[i + 1][w].value += 1.0f * layerBiases[i][b].outWeights[w];
            }
        }
    }
}

vector<float> NeuralNetwork::predict(vector<float> inputs){
    feedForward(inputs);

    vector<float> result;
    int lastLayerSize = layerNodes[layerCount - 1].size();

    for(int i = 0; i < lastLayerSize; i++){
        result.push_back(layerNodes[layerCount - 1][i].value);
    }
    return result;
}

void NeuralNetwork::calculateDerivatives(vector<float> outputErrors){
    // with outputErrors as actual - target
    int finalLayerCount = layerNodes[layerCount - 1].size();
    for(int i = 0; i < finalLayerCount; i++){
        layerNodes[layerCount - 1][i].derivativeErrorValue = derivative(layerNodes[layerCount - 1][i].value) * outputErrors[i];
    }
    // work backwards
    for(int i = layerCount - 2; i > -1; i--){
        int currentLayerCount = layerNodes[i].size();

        for(int n = 0; n < currentLayerCount; n++){
            int outWeightCount = layerNodes[i][n].outWeights.size();

            float valueMultiplier = derivative(layerNodes[i][n].value);
            if(i == 0){
                valueMultiplier = layerNodes[i][n].value;
            }

            for(int w = 0; w < outWeightCount; w++){
                layerNodes[i][n].derivativeErrorValue += valueMultiplier * layerNodes[i][n].outWeights[w] * layerNodes[i + 1][w].derivativeErrorValue;
            }
        }

    }
}

void NeuralNetwork::adjustWeights(float lr, float momentum){
    for(int i = 0; i < layerCount; i++){
        int nodeCount = layerNodes[i].size();

        for(int n = 0; n < nodeCount; n++){
            int weightCount = layerNodes[i][n].outWeights.size();

            for(int w = 0; w < weightCount; w++){
                float newDelta = layerNodes[i][n].value * layerNodes[i + 1][w].derivativeErrorValue * lr;
                layerNodes[i][n].outWeights[w] += newDelta;

                layerNodes[i][n].outWeights[w] += layerNodes[i][n].previousDeltas[w] * momentum;
                layerNodes[i][n].previousDeltas[w] = newDelta;
            }
        }

        int biasCount = layerBiases[i].size();
        for(int b = 0; b < biasCount; b++){
            int outWeightCount = layerBiases[i][b].outWeights.size();

            for(int w = 0; w < outWeightCount; w++){
                float newDelta = 1.0f * layerNodes[i + 1][w].derivativeErrorValue * lr;
                layerBiases[i][b].outWeights[w] += newDelta;

                layerBiases[i][b].outWeights[w] += layerBiases[i][b].previousDeltas[w] * momentum;
                layerBiases[i][b].previousDeltas[w] = newDelta;
            }
        }
    }
}

void NeuralNetwork::resetDerivativesAndResults(){
    for(int i = 0; i < layerCount; i++){
        int nodeCount = layerNodes[i].size();

        for(int n = 0; n < nodeCount; n++){
            layerNodes[i][n].derivativeErrorValue = 0.0f;
            layerNodes[i][n].value = 0.0f;
        }
    }
}

void NeuralNetwork::train(vector<vector<float>> trainInputs, vector<vector<float>> trainOutputs, int epochs, float lr, float momentum){
    int trainDataCount = trainInputs.size();
    int outputCount = trainOutputs[0].size();

    vector<int> trainIndexes;
    for(int i = 0; i < trainDataCount; i++){
        trainIndexes.push_back(i);
    }

    for(int epoch = 0; epoch < epochs; epoch++){
        cout << ((float) (epoch + 1) / (float) epochs) * 100.0f << "%" << endl;
        random_shuffle(trainIndexes.begin(), trainIndexes.end());
        for(int t = 0; t < trainDataCount; t++){
            int currentIndex = trainIndexes[t];
            vector<float> result = predict(trainInputs[currentIndex]);
            
            vector<float> errors;
            float totalError = 0.0f;
            for(int e = 0; e < outputCount; e++){
                float error = trainOutputs[currentIndex][e] - result[e];
                totalError += error;

                errors.push_back(error);
            }
            cout << t << " / " << trainDataCount << " " << totalError << endl;

            calculateDerivatives(errors);
            adjustWeights(lr, momentum);
        }
    }
}

void NeuralNetwork::runTests(vector<vector<float>> inputs){
    int count = inputs.size();
    for(int i = 0; i < count; i++){
        cout << vectorToString(predict(inputs[i])) << endl;
    }
}

void NeuralNetwork::saveWeightsToFile(string directory){
    string fileNameNodes = directory + "nodeWeights.txt";
    string fileNameBias = directory + "biasWeights.txt";

    vector<string> newLinesNodes;
    for(int i = 0; i < layerCount; i++){
        int currentNodeCount = layerNodes[i].size();

        for(int n = 0; n < currentNodeCount; n++){
            string currentNodeWeights = "";

            int outWeightCount = layerNodes[i][n].outWeights.size();
            for(int w = 0; w < outWeightCount; w++){
                currentNodeWeights += to_string(layerNodes[i][n].outWeights[w]) + ",";
            }

            int stringLength = currentNodeWeights.size();
            if(stringLength == 0){
                continue;
            }

            currentNodeWeights.pop_back();
            newLinesNodes.push_back(currentNodeWeights);
        }
    }

    vector<string> newLinesBias;
    for(int i = 0; i < layerCount; i++){
        int currentBiasCount = layerBiases[i].size();

        for(int b = 0; b < currentBiasCount; b++){
            string currentBiasWeights = "";

            int outWeightCount = layerBiases[i][b].outWeights.size();
            for(int w = 0; w < outWeightCount; w++){
                currentBiasWeights += to_string(layerBiases[i][b].outWeights[w]) + ",";
            }

            currentBiasWeights.pop_back();
            newLinesBias.push_back(currentBiasWeights);
        }
    }

    // write to file
    writeToFile(fileNameNodes.c_str(), newLinesNodes);
    writeToFile(fileNameBias.c_str(), newLinesBias);
}

void NeuralNetwork::loadWeightsFromFile(string directory){
    string fileNameNodes = directory + "nodeWeights.txt";
    string fileNameBias = directory + "biasWeights.txt";

    vector<string> nodeLines = readFile(fileNameNodes.c_str());
    vector<string> biasLines = readFile(fileNameBias.c_str());

    // nodes
    int currentNodesSeen = 0;
    for(int i = 0; i < layerCount; i++){
        int currentNodeCount = layerNodes[i].size();

        for(int n = 0; n < currentNodeCount; n++){
            vector<string> currentLineData = splitStringByCharacter(nodeLines[currentNodesSeen], ',');

            int weightCount = layerNodes[i][n].outWeights.size();
            for(int w = 0; w < weightCount; w++){
                layerNodes[i][n].outWeights[w] = stof(currentLineData[w]);
            }

            currentNodesSeen += 1;
        }
    }

    // bias
    currentNodesSeen = 0;
    for(int i = 0; i < layerCount; i++){
        int currentBiasCount = layerBiases[i].size();

        for(int n = 0; n < currentBiasCount; n++){
            vector<string> currentLineData = splitStringByCharacter(biasLines[currentNodesSeen], ',');

            int weightCount = layerBiases[i][n].outWeights.size();
            for(int w = 0; w < weightCount; w++){
                layerBiases[i][n].outWeights[w] = stof(currentLineData[w]);
            }

            currentNodesSeen += 1;
        }
    }
}