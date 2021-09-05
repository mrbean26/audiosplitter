#include "Headers/NeuralNetwork.h"

#include <iostream>
using namespace std;

#include "Headers/files.h"
#include "Headers/audio.h"

int main() {
	srand(time(NULL));
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

	vector<vector<float>> inputSet = generateInputs(audioConfig);
	vector<vector<float>> outputSet = generateOutputs(audioConfig);

	int inputSize = inputSet[0].size();
	int outputSize = outputSet[0].size();

	vector<int> layers = { inputSize, 128, 128, 128, 128, 128, 128, 128, 128, outputSize };
	vector<int> biases = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	vector<int> activations = { SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID };

	// Natural Selection
	NeuralNetwork::standardTrainConfig trainConfig = NeuralNetwork::standardTrainConfig();
	trainConfig.trainType = GRADIENT_DESCENT;

	trainConfig.trainInputs = inputSet;
	trainConfig.trainOutputs = outputSet;

	trainConfig.useDropout = true;
	trainConfig.nodeBiasDropoutProbability = 25;

	trainConfig.epochs = 1000;
	trainConfig.batchSize = 25;

	trainConfig.learningRate = 1.5f;
	trainConfig.momentum = 0.2f;
	trainConfig.learningRateType = CYCLICAL_LEARNING_RATE;

	NeuralNetwork newNetwork = NeuralNetwork(layers, biases, activations);
	newNetwork.train(trainConfig);

	system("pause");
	return 0;
}