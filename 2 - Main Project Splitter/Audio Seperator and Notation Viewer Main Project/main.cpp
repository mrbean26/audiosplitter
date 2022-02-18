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
		0.1f, // binary mask threshold

		false, // use noise prediction
	};

	// Train Config
	NeuralNetwork::standardTrainConfig newConfig = NeuralNetwork::standardTrainConfig();

	newConfig.trainType = BATCH_GRADIENT_DESCENT;
	newConfig.epochs = 3;

	newConfig.gradientDescent.learningRateType = CYCLICAL_LEARNING_RATE;
	newConfig.learningRate = 1.0f;
	newConfig.momentum = 0.25f;

	newConfig.trainInputs = generateInputs(audioConfig);
	newConfig.trainOutputs = generateOutputs(audioConfig);
	newConfig.gradientDescent.datasetAudioConfig = audioConfig;

	// Train Network
	vector<int> nodes = { 256, 448, 256, 64, 768, 448, 384, 256, 256, 448, 32 };
	vector<int> bias(nodes.size(), 1);
	vector<int> activations(nodes.size(), SIGMOID);

	NeuralNetwork vocalNetwork = NeuralNetwork(nodes, bias, activations);
	vocalNetwork.train(newConfig);

	createOutputTestTrack(vocalNetwork, audioConfig);

	system("pause");
	return 0;
}