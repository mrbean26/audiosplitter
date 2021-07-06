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
		{0.0f},
		{1.0f},
		{1.0f},
		{0.0f}
	};

	NeuralNetwork newNetwork = NeuralNetwork({ 2, 2, 1 }, { 1, 1, 0 }, "sigmoid");

	NeuralNetwork::standardTrainConfig newConfig = NeuralNetwork::standardTrainConfig();
	newConfig.trainInputs = inputSet;
	newConfig.trainOutputs = outputSet;
	newConfig.trainType = STOCHASTIC_GRADIENT_DESCENT;

	newNetwork.train(newConfig);
	newNetwork.runTests(inputSet);

	system("pause");
	return 0;
}