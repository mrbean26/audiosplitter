#include "Headers/NeuralNetwork.h"

#include <iostream>
using namespace std;

#include "Headers/files.h"
#include "Headers/audio.h"

int main() {
	audioFileConfig audioConfig = {
		2048, // samples per chunk
		2048, // samples per overlap

		128, // frequency res
		4, // chunk border

		1, // start file index
		1 // song count
	};

	// Train Network - One Song Training
	vector<vector<float>> inputSet = generateInputs(audioConfig);
	vector<vector<float>> outputSet = generateOutputs(audioConfig);

	int iterationsPerEach = 2;
	int minimumLayers = 4;

	NeuralNetwork::trainSeveralConfigurations(audioConfig, inputSet, outputSet, 1000, minimumLayers, iterationsPerEach, outputSet[0].size(), 0.2f, 0.05f);

	// Test with first test songs
	//createOutputTestTrack(network, audioConfig);

	system("pause");
	return 0;
}