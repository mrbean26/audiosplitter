#include "Headers/NeuralNetwork.h"

#include <iostream>
using namespace std;

#include "Headers/files.h"
#include "Headers/audio.h"

#include "Headers/matrices.h"

int main() {
	audioFileConfig audioConfig = {
		2048, // samples per chunk
		2048, // samples per overlap

		128, // frequency res
		4, // chunk border

		1, // start file index
		1 // song count
	};

	// Train Network - One Song Training
	vector<vector<float>> inputSet = generateInputs(audioConfig);
	vector<vector<float>> outputSet = generateOutputs(audioConfig);

	inputSet = {
		{0.0f, 0.0f},
		{1.0f, 0.0f},
		{0.0f, 1.0f},
		{1.0f, 1.0f}
	};

	outputSet = {
		{0.0f, 0.0f},
		{1.0f, 1.0f},
		{1.0f, 1.0f},
		{0.0f, 0.0f}
	};

	vector<int> layers = { 2, 3, 3, 2 };
	vector<int> biases = { 1, 1, 1, 1, 1, 1 };

	NeuralNetwork newNetwork = NeuralNetwork(layers, biases, "sigmoid");
	
	NeuralNetwork::standardTrainConfig newConfig = NeuralNetwork::standardTrainConfig();
	newConfig.trainInputs = inputSet;
	newConfig.epochs = 1000;
	newConfig.entireBatchEpochIntervals = 1000;
	newConfig.trainOutputs = outputSet;
	newConfig.batchSize = 100;
	newConfig.useCyclicalLearningRateAndMomentum = true;

	newConfig.trainType = LEVENBERG_MARQUARDT;

	newNetwork.train(newConfig);
	newNetwork.runTests(inputSet);

	system("pause");
	return 0;
}