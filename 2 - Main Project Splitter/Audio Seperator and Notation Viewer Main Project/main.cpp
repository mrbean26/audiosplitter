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
		4, // song count

		2.5f, // spectrogram emphasis, no emphasis = 1.0f

		false, // use binary mask for output
		0.025f, // binary mask threshold
	};
	
	// Train Config
	NeuralNetwork::standardTrainConfig newConfig = NeuralNetwork::standardTrainConfig();
	
	newConfig.trainType = STOCHASTIC_GRADIENT_DESCENT;
	newConfig.epochs = 200;
	
	newConfig.gradientDescent.learningRateType = CYCLICAL_LEARNING_RATE;
	newConfig.learningRate = 1.0f;
	newConfig.momentum = 0.25f;

	newConfig.gradientDescent.useAllSongDataset = true;
	newConfig.gradientDescent.batchSize = 100 * 200; // 100 songs * 200 per song
	newConfig.gradientDescent.datasetAudioConfig = audioConfig;

	// Train Network
	vector<int> nodes = { 256, 384, 128, 128, 128, 512, 640, 384, 384, 384, 32 };
	vector<int> bias = { 2, 1, 3, 1, 3, 3, 3, 2, 2, 4, 0 };
	vector<int> activations = { SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID };
	
	NeuralNetwork vocalNetwork = NeuralNetwork(nodes, bias, activations);
	vocalNetwork.train(newConfig);

	createOutputTestTrack(vocalNetwork, audioConfig);	
	system("pause");

	return 0;
}