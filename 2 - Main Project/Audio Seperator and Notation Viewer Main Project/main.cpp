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

		64, // frequency res
		4, // chunk border

		1, // start file index
		1, // song count

		2.5f, // spectrogram emphasis, no emphasis = 1.0f

		false, // use binary mask for output
		0.025f, // binary mask threshold
	};

	// Train Network - One Song Training
	vector<vector<float>> inputSet = generateInputs(audioConfig);
	vector<vector<float>> outputSet = generateOutputs(audioConfig);
	
	int inputSize = inputSet[0].size();
	int outputSize = outputSet[0].size();

	vector<int> layers = { inputSize, 128, 128, 128, 128, outputSize };
	vector<int> biases = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	vector<int> activations = { SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID };
	// natural selection


	NeuralNetwork::standardTrainConfig newConfig = NeuralNetwork::standardTrainConfig();
	newConfig.trainInputs = inputSet;
	newConfig.trainOutputs = outputSet;

	newConfig.epochs = 125;

	newConfig.population = 300;
	newConfig.parentCount = 3;

	newConfig.lowestInitialisedWeight = -10.0f;
	newConfig.highestInitialisedWeight = 10.0f;


	NeuralNetwork bestNetwork = NeuralNetwork::trainNaturalSelectionMethod(newConfig, layers, biases, activations);

	/*
	 
	BACKPROPAGATION ALGORITHM 
	
	NeuralNetwork newNetwork = NeuralNetwork(layers, biases, activations);
	
	NeuralNetwork::standardTrainConfig newConfig = NeuralNetwork::standardTrainConfig();
	newConfig.trainInputs = inputSet;
	newConfig.trainOutputs = outputSet;

	newConfig.epochs = 25000;
	newConfig.learningRate = 0.75f;
	newConfig.momentum = 0.125f;
	newConfig.learningRateType = CYCLICAL_LEARNING_RATE;

	newConfig.entireBatchEpochIntervals = 1000;
	newConfig.batchSize = 400;

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
	*/

	system("pause");
	return 0;
}