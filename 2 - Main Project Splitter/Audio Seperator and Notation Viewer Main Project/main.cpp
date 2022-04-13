#include "Headers/NeuralNetwork.h"

#include <iostream>
using namespace std;

#include "Headers/files.h"
#include "Headers/audio.h"

audioFileConfig getAudioConfig() {
	audioFileConfig audioConfig = {
		512, // samples per chunk
		64, // samples per overlap

		128, // frequency res
		12, // chunk border

		1, // start file index
		1, // song count

		0.5f, // spectrogram emphasis, no emphasis = 1.0f

		true, // use binary mask for output
		0.1f, // binary mask threshold

		false, // use noise prediction

		true, // use mel scale
	};

	return audioConfig;
}

NeuralNetwork::standardTrainConfig getTrainConfig() {
	NeuralNetwork::standardTrainConfig newConfig = NeuralNetwork::standardTrainConfig();

	newConfig.trainType = BATCH_GRADIENT_DESCENT;
	newConfig.epochs = 50;

	newConfig.gradientDescent.learningRateType = CYCLICAL_LEARNING_RATE;
	newConfig.learningRate = 0.75f;
	newConfig.momentum = 0.1f;
	
	// 1st song training
	newConfig.gradientDescent.useAllSongDataset = true;
	newConfig.gradientDescent.datasetAudioConfig = getAudioConfig();
	newConfig.gradientDescent.allSongDatasetStart = 0;
	newConfig.gradientDescent.allSongDatasetEnd = 10;
	newConfig.gradientDescent.batchSize = 500;
	
	newConfig.gradientDescent.useThreading = false;

	return newConfig;
}

int main() {
	srand(time(NULL));
	
	// Train Config
	audioFileConfig audioConfig = getAudioConfig();
	NeuralNetwork::standardTrainConfig trainConfig = getTrainConfig();
	
	vector<int> nodes = { 1600, 1600, 800, 800, 400, 400, 200, 64  };
	vector<int> bias(nodes.size(), 1);
	vector<int> activations(nodes.size(), SIGMOID);

	// Train
	NeuralNetwork vocalsNetwork = NeuralNetwork(nodes, bias, activations);
	vocalsNetwork.train(trainConfig);

	createOutputTestTrack(vocalsNetwork, audioConfig);

	system("pause");
	return 0;
}