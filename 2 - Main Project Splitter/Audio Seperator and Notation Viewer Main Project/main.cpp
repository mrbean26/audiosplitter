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

		2048, // frequency res
		12, // chunk border

		1, // start file index
		1, // song count

		1.0f, // spectrogram emphasis, no emphasis = 1.0f

		true, // use binary mask for output
		0.025f, // binary mask threshold
	};

	// Train Config
	NeuralNetwork::standardTrainConfig newConfig = NeuralNetwork::standardTrainConfig();
	
	newConfig.trainType = BATCH_GRADIENT_DESCENT;
	newConfig.epochs = 25;
	newConfig.learningRate = 1.0f;
	newConfig.momentum = 0.25f;

	vector<vector<float>> outputs = generateOutputs(audioConfig);
	vector<int16_t> testTrackOutputSamples = vocalSamples("inputs/1.mp3", outputs, audioConfig);
	writeToWAV("testTrackOutput.wav", testTrackOutputSamples);

	system("pause");
	return 0;
}