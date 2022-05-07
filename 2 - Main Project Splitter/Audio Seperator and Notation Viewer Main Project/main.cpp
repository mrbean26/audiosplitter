#include "Headers/NeuralNetwork.h"

#include <iostream>
using namespace std;

#include "Headers/files.h"
#include "Headers/audio.h"

audioFileConfig getAudioConfig() {
	audioFileConfig audioConfig = {
		256, // samples per chunk
		256, // samples per overlap

		128, // frequency res
		4, // chunk border

		1, // start file index
		100, // song count

		2.0f, // spectrogram emphasis, no emphasis = 1.0f

		true, // use binary mask for output
		0.05f, // binary mask threshold

		false, // use noise prediction
		true, // use mel scale

		true, // skip chunk overlap

		true, // single output value
		0.1f, // percentage of chunk needed to be 1

		20, // chunk size
		12, // required chunk count
	};

	return audioConfig;
}
NeuralNetwork::standardTrainConfig getTrainConfig() {
	NeuralNetwork::standardTrainConfig newConfig = NeuralNetwork::standardTrainConfig();

	newConfig.trainType = GRADIENT_DESCENT;
	newConfig.epochs = 300;

	newConfig.gradientDescent.learningRateType = FIXED_LEARNING_RATE;
	newConfig.learningRate = 0.02f;
	newConfig.momentum = 0.05f;

	// All song training
	/*
	newConfig.gradientDescent.useAllSongDataset = true;
	newConfig.gradientDescent.datasetAudioConfig = getAudioConfig();
	newConfig.gradientDescent.allSongDatasetStart = 0;
	newConfig.gradientDescent.allSongDatasetEnd = 15;
	newConfig.gradientDescent.batchSize = 2000;
	newConfig.gradientDescent.datasetRefreshInterval = 1000;
	newConfig.gradientDescent.useThreading = false; */

	vector<vector<float>> i = generateInputs(getAudioConfig());
	vector<vector<float>> o = generateOutputs(getAudioConfig());
	
	int miniBatchSize = i.size() / 30000;
	for (int j = 0; j < 30000; j++) {
		int newIndex = (j * miniBatchSize) + (rand() % miniBatchSize);

		newConfig.trainInputs.push_back(i[newIndex]);
		newConfig.trainOutputs.push_back(o[newIndex]);
	}

	return newConfig;
}

int main() {
	srand(time(NULL));
	
	// Train Config
	audioFileConfig audioConfig = getAudioConfig();
	//NeuralNetwork::standardTrainConfig trainConfig = getTrainConfig();
	
	// Tests
	testNetworkInputsToImage(audioConfig);
	testNetworkOutputsToImage(audioConfig);
	testTrainOutputs(audioConfig);
	inputTrackSpectrogramToImage(audioConfig);
	system("explorer D:\\Projects\\Audio Splitter App\\2 - Main Project Splitter\\Audio Seperator and Notation Viewer Main Project\\_Testing\\");

	// Network & Trainig
	vector<int> nodes = { 576, 400, 300, 300, 200, 200, 200, 150, 100, 100, 100, 1 };
	vector<int> bias(nodes.size(), 1);
	vector<int> activations(nodes.size(), SIGMOID);

	NeuralNetwork vocalsNetwork = NeuralNetwork(nodes, bias, activations);
	//vocalsNetwork.train(trainConfig);
	vocalsNetwork.loadWeightsFromFile("w_tmp/");
	createOutputTestTrack(vocalsNetwork, audioConfig);
	//vocalsNetwork.saveWeightsToFile("w_tmp/");

	system("pause");
	return 0;
}