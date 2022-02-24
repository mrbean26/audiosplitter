#include "Headers/NeuralNetwork.h"

#include <iostream>
using namespace std;

#include "Headers/files.h"
#include "Headers/audio.h"

int main() {
	srand(time(NULL));
	audioFileConfig audioConfig = {
		512, // samples per chunk
		512, // samples per overlap

		512, // frequency res
		12, // chunk border

		1, // start file index
		1, // song count

		2.5f, // spectrogram emphasis, no emphasis = 1.0f

		false, // use binary mask for output
		0.025f, // binary mask threshold
	};

	// Train Config
	NeuralNetwork::standardTrainConfig newConfig = NeuralNetwork::standardTrainConfig();

	newConfig.trainType = BATCH_GRADIENT_DESCENT;
	newConfig.epochs = 25;

	newConfig.gradientDescent.learningRateType = CYCLICAL_LEARNING_RATE;
	newConfig.learningRate = 1.0f;
	newConfig.momentum = 0.25f;

	newConfig.gradientDescent.useAllSongDataset = true;
	newConfig.gradientDescent.batchSize = 100 * 50; // 100 songs * 50 per song
	newConfig.gradientDescent.datasetAudioConfig = audioConfig;

	// Train Network
	vector<int> nodes = { 6144, 6144, 6144, 6144, 6144, 6144, 2048, 2048, 2048, 2048, 2048, 2048, 682, 682, 682, 682, 682, 256 };
	vector<int> bias(nodes.size(), 1);
	vector<int> activations(nodes.size(), SIGMOID);

	NeuralNetwork vocalsNetwork = NeuralNetwork(nodes, bias, activations);
	vocalsNetwork.train(newConfig);
	//vocalNetwork.loadWeightsFromFile("vocalsOutputWeights/");
	vocalsNetwork.saveWeightsToFile("weights/vocalOutputWeights/");
	createOutputTestTrack(vocalsNetwork, audioConfig);

	system("pause");
	return 0;
}