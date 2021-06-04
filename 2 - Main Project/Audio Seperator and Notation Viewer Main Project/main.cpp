#include "Headers/NeuralNetwork.h"

#include <iostream>
using namespace std;

#include "Headers/files.h"
#include "Headers/audio.h"

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

	vector<int> layers = { inputSize, outputSize * 2, outputSize * 2, outputSize * 2, outputSize, };
	vector<int> biases = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	NeuralNetwork network = NeuralNetwork(layers, biases, "tanh");
	
	standardTrainConfig trainingConfig = {
		inputSet,
		outputSet,

		1000, // Epochs

		1.0f, // LR
		0.25f, // Momentum

		true, // Use Cyclical Learning Rate?

		false, // Use Weight Decay ?
		0.99f, // Weight Decay Multiplier

		0.5f, // RPROP Weight Decrease
		1.2f, // RPROP Weight Increase

		false, // Use Dropout ?
		4, // 1 in "x" Random Nodes / Biases dropped
	};
	vector<float> trainingErrors = network.train(trainingConfig);

	outputImageConfig imageConfig{
		trainingErrors,

		1000, // error res
		512, // error range

		network,
		audioConfig,
		trainingConfig,

		false, // Use fixed scale?
		500.0f, // max scale on fixed scale
		0.0f, // min scale on fixed scale
	};
	
	writeToImage(imageConfig);

	// Test with first test songs
	createOutputTestTrack(network, audioConfig);

	system("pause");
	return 0;
}