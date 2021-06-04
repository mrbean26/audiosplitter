#ifndef FILES_H
#define FILES_H

#include "Headers/NeuralNetwork.h"

#include <vector>
using namespace std;

struct audioFileConfig {
	int samplesPerChunk = 2048;
	int samplesPerOverlap = 2048;

	int frequencyResolution = 128;
	int chunkBorder = 4;

	int startFileIndex = 1;
	int songCount = 1;
};

vector<vector<float>> generateInputs(audioFileConfig config);
vector<vector<float>> generateOutputs(audioFileConfig config);

vector<vector<float>> addCharacterToImage(vector<vector<float>> data, int character, int xMidpoint, int yMidpoint);

struct outputImageConfig {
	vector<float> errors;

	int errorResolution;
	int errorRange;

	NeuralNetwork network;
	audioFileConfig audioConfig;
	standardTrainConfig trainConfig;

	bool useFixedScale = true;
	float fixedMax = 5000.0f;
	float fixedMin = 0.0f;
};

vector<vector<float>> addCharacterToImage(vector<vector<float>> data, int character, int xMidpoint, int yMidpoint);
void writeToImage(outputImageConfig config);

void createOutputTestTrack(NeuralNetwork network, audioFileConfig config);

#endif