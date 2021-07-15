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

	vector<int> layers = { 512, 64 };
	vector<int> biases = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	NeuralNetwork newNetwork = NeuralNetwork(layers, biases, "tanh");
	
	NeuralNetwork::standardTrainConfig newConfig = NeuralNetwork::standardTrainConfig();
	newConfig.trainInputs = inputSet;
	newConfig.trainOutputs = outputSet;

	newConfig.epochs = 1000;
	newConfig.learningRate = 1.0f;
	newConfig.momentum = 0.25f;
	newConfig.useCyclicalLearningRateAndMomentum = true;

	newConfig.entireBatchEpochIntervals = 1000;
	newConfig.batchSize = 300;

	newConfig.trainType = STOCHASTIC_GRADIENT_DESCENT;

	vector<float> errors = newNetwork.train(newConfig);
	createOutputTestTrack(newNetwork, audioConfig);

	outputImageConfig imageConfig = outputImageConfig{
		errors,

		1000,
		512,

		newNetwork,
		audioConfig,
		newConfig,

		true,
		1000.0f,
		0.0f,
	};

	writeToImage(imageConfig);

	system("pause");
	return 0;
}