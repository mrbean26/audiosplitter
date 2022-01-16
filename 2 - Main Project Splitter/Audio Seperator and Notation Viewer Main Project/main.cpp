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
	
	newConfig.gradientDescent.learningRateType = DECREASING_LEARNING_RATE;
	newConfig.learningRate = 1.0f;
	newConfig.momentum = 0.25f;

	newConfig.gradientDescent.datasetRefreshInterval = 5;
	newConfig.gradientDescent.useAllSongDataset = true;
	newConfig.gradientDescent.batchSize = 100 * 200; // 100 songs * 200 per song
	newConfig.gradientDescent.datasetAudioConfig = audioConfig;

	// Train Network
	vector<int> nodes = { 256, 448, 256, 64, 768, 448, 384, 256, 256, 448, 32 };
	vector<int> bias = { 3, 5, 3, 5, 5, 1, 2, 3, 1, 5, 0 };
	vector<int> activations = { SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID };
	
	NeuralNetwork bassNetwork = NeuralNetwork(nodes, bias, activations);
	bassNetwork.train(newConfig);

	bassNetwork.saveWeightsToFile("bassOutputWeights.dat");
	createOutputTestTrack(bassNetwork, audioConfig);	

	system("pause");
	return 0;
}