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

	int inputSize = inputSet[0].size();
	int outputSize = outputSet[0].size();

	vector<int> layers = { inputSize, outputSize * 4, outputSize * 3, outputSize * 2, outputSize };
	vector<int> biases = { 1, 1, 1, 1, 1, 1, 1, 1 };

	NeuralNetwork newNetwork = NeuralNetwork(layers, biases, "sigmoid");
	
	NeuralNetwork::standardTrainConfig newConfig = NeuralNetwork::standardTrainConfig();
	newConfig.trainInputs = inputSet;
	newConfig.epochs = 10000;
	newConfig.trainOutputs = outputSet;
	newConfig.batchSize = 200;

	newConfig.trainType = STOCHASTIC_GRADIENT_DESCENT;

	newNetwork.train(newConfig);
	
	system("pause");
	return 0;
}