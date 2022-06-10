#include "Headers/NeuralNetwork.h"

#include <iostream>
using namespace std;

#include "Headers/files.h"
#include "Headers/audio.h"

audioFileConfig getAudioConfig() {
	audioFileConfig audioConfig = {
		1024, // samples per chunk
		1024, // samples per overlap

		128, // frequency res
		8, // chunk border

		1, // start file index
		3, // song count

		2.0f, // spectrogram emphasis, no emphasis = 1.0f

		true, // use binary mask for output
		0.1f, // binary mask threshold

		false, // use noise prediction
		true, // use mel scale

		true, // skip chunk overlap

		false, // single output value
		0.1f, // percentage of chunk needed to be 1

		20, // chunk size
		12, // required chunk count
	};

	return audioConfig;
}
NeuralNetwork::standardTrainConfig getTrainConfig() {
	NeuralNetwork::standardTrainConfig newConfig = NeuralNetwork::standardTrainConfig();

	newConfig.trainType = GRADIENT_DESCENT;
	newConfig.epochs = 1000;

	newConfig.gradientDescent.learningRateType = FIXED_LEARNING_RATE;
	newConfig.learningRate = 0.01f;
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

	pair<vector<vector<float>>, vector<vector<float>>> allSongDataset = generateAllSongDataSet(getAudioConfig(), 1000, 0, 100);

	newConfig.trainInputs = allSongDataset.first;
	newConfig.trainOutputs = allSongDataset.second;
	cout << newConfig.trainInputs.size() << endl;

	return newConfig;
}
void runDataTests(audioFileConfig audioConfig) {
	testNetworkInputsToImage(audioConfig);
	testNetworkOutputsToImage(audioConfig);
	testTrainOutputs(audioConfig);
	inputTrackSpectrogramToImage(audioConfig);
	system("explorer D:\\Projects\\Audio Splitter App\\2 - Main Project Splitter\\Audio Seperator and Notation Viewer Main Project\\_Testing\\");
}

int main() {
	srand(time(NULL));

	// Train Config
	audioFileConfig audioConfig = getAudioConfig();
	//NeuralNetwork::standardTrainConfig trainConfig = getTrainConfig();
	runDataTests(audioConfig);
	
	// Network & Trainig
	vector<int> nodes = { 1088, 750, 500, 300, 100, 64 };
	vector<int> bias(nodes.size(), 1);
	vector<int> activations(nodes.size(), SIGMOID);

	NeuralNetwork vocalsNetwork = NeuralNetwork(nodes, bias, activations);
	//vocalsNetwork.train(trainConfig);
	
	vocalsNetwork.loadWeightsFromFile("_trained_weights/1st_proper_train/");
	createOutputTestTrack(vocalsNetwork, audioConfig, "test_tracks/californication.mp3");
	//vocalsNetwork.saveWeightsToFile("_trained_weights/1st_proper_train/");

	system("pause");
	return 0;
}