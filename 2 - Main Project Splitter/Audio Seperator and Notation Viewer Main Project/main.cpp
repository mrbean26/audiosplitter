#include "Headers/NeuralNetwork.h"

#include <iostream>
using namespace std;

#include "Headers/files.h"
#include "Headers/audio.h"

audioFileConfig getAudioConfig() {
	audioFileConfig audioConfig = {
		512, // samples per chunk
		512, // samples per overlap

		512, // frequency res
		2, // chunk border

		1, // start file index
		1, // song count

		2.0f, // spectrogram emphasis, no emphasis = 1.0f

		true, // use binary mask for output
		0.1f, // binary mask threshold

		false, // use noise prediction
		true, // use mel scale

		true, // skip chunk overlap
	};

	return audioConfig;
}
NeuralNetwork::standardTrainConfig getTrainConfig() {
	NeuralNetwork::standardTrainConfig newConfig = NeuralNetwork::standardTrainConfig();

	newConfig.trainType = BATCH_GRADIENT_DESCENT;
	newConfig.epochs = 10000;

	newConfig.gradientDescent.learningRateType = FIXED_LEARNING_RATE;
	newConfig.learningRate = 0.8f;
	newConfig.momentum = 0.05f;
	
	// ALL song training
	
	newConfig.gradientDescent.useAllSongDataset = true;
	newConfig.gradientDescent.datasetAudioConfig = getAudioConfig();
	newConfig.gradientDescent.allSongDatasetStart = 0;
	newConfig.gradientDescent.allSongDatasetEnd = 10;
	newConfig.gradientDescent.batchSize = 2000;

	newConfig.gradientDescent.datasetRefreshInterval = 1000;

	newConfig.gradientDescent.useThreading = false;

	return newConfig;
}

int main() {
	srand(time(NULL));
	
	// Train Config
	audioFileConfig audioConfig = getAudioConfig();
	NeuralNetwork::standardTrainConfig trainConfig = getTrainConfig();
	
	// Tests
	testNetworkInputsToImage(audioConfig);
	testNetworkOutputsToImage(audioConfig);
	testTrainOutputs(audioConfig);
	inputTrackSpectrogramToImage(audioConfig);
	system("explorer D:\\Projects\\Audio Splitter App\\2 - Main Project Splitter\\Audio Seperator and Notation Viewer Main Project\\_Testing\\");

	// Network & Trainig
	vector<int> nodes = { 80, 40, 40, 16 };
	vector<int> bias(nodes.size(), 1);
	vector<int> activations(nodes.size(), SIGMOID);

	NeuralNetwork vocalsNetwork = NeuralNetwork(nodes, bias, activations);
	//vocalsNetwork.train(trainConfig);
	//createOutputTestTrack(vocalsNetwork, audioConfig);

	system("pause");
	return 0;
}