#pragma once

#include <vector>
#include <string>
#include <iostream>

using namespace std;

namespace activations {
	float sigmoid(float x);
	float tanh(float x);
	float relu(float x);
	float leaky_relu(float x);
}

namespace derivatives {
	float sigmoid(float x);
	float tanh(float x);
	float relu(float x);
	float leaky_relu(float x);
}

string vectorToString(vector<float> used);
float activate(float x);
float derive(float x);
extern const char* activationType;

class NeuralNetwork {
private:
	vector<vector<float>> network;
	vector<vector<float>> previousDeltas;

	int layerCount;
	vector<int> nodeCounts;
	vector<int> biasCounts;
public:
	NeuralNetwork(vector<int> layerNodeCount,
		vector<int> layerBiasCount, const char* activation);

	vector<float> feedForward(vector<float> inputs);
	void train(int epochs, float learningRate, float momentum, 
		vector<vector<float>> trainInputs, vector<vector<float>> trainOutputs);

	void setDerivativesAndDeltaWeights(vector<float> errors, float learningRate, float momentum);
	void resetValues();
	void runTests(vector<vector<float>> testInputs);
};