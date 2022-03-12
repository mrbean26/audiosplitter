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
		4, // song count

		1.0f, // spectrogram emphasis, no emphasis = 1.0f

		false, // use binary mask for output
		0.025f, // binary mask threshold
	};

	// Train Config
	NeuralNetwork::standardTrainConfig newConfig = NeuralNetwork::standardTrainConfig();
	
	newConfig.trainType = BATCH_GRADIENT_DESCENT;
	newConfig.epochs = 25;

	newConfig.gradientDescent.learningRateType = ADAM_LEARNING_RATE;
	newConfig.learningRate = 0.001f;
	newConfig.gradientDescent.betaOne = 0.9f;
	newConfig.gradientDescent.betaTwo = 0.999f;
	newConfig.gradientDescent.epsillon = 0.00000001f;

	newConfig.gradientDescent.useAllSongDataset = true;
	newConfig.gradientDescent.batchSize = 100 * 50; // 100 songs * 50 per song
	newConfig.gradientDescent.datasetAudioConfig = audioConfig;
	newConfig.gradientDescent.useThreading = false;

	// Train Network
	//vector<int> nodes = { 6144, 6144, 6144, 6144, 6144, 6144, 2048, 2048, 2048, 2048, 2048, 2048, 682, 682, 682, 682, 682, 256 };
	vector<int> nodes = { 6144, 3072, 1536, 768, 256 };

	vector<int> bias(nodes.size(), 1);
	vector<int> activations(nodes.size(), SIGMOID);

	NeuralNetwork vocalsNetwork = NeuralNetwork(nodes, bias, activations);
	vocalsNetwork.train(newConfig);
	//vocalNetwork.loadWeightsFromFile("vocalsOutputWeights/");
	//vocalsNetwork.saveWeightsToFile("weights/vocalOutputWeightsNoEmphasis/");
	createOutputTestTrack(vocalsNetwork, audioConfig);

	system("pause");
	return 0;
}