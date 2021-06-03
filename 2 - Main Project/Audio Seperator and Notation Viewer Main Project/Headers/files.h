#ifndef FILES_H
#define FILES_H

#include "Headers/NeuralNetwork.h"

#include <vector>
using namespace std;

vector<vector<float>> generateInputs(int samplesPerChunk, int samplesPerOverlap, int frequencyResolution, int chunksPerInputHalf, int startFileIndex, int endIndex);
vector<vector<float>> generateOutputs(int samplesPerChunk, int samplesPerOverlap, int frequencyResolution, int chunksPerInputHalf, int startFileIndex, int endIndex);

vector<vector<float>> addCharacterToImage(vector<vector<float>> data, int character, int xMidpoint, int yMidpoint);

struct outputImageConfig {
	vector<float> errors;

	int errorResolution;
	int errorRange;

	NeuralNetwork network;

	bool useFixedScale = true;
	float fixedMax = 5000.0f;
	float fixedMin = 0.0f;
};

vector<vector<float>> addCharacterToImage(vector<vector<float>> data, int character, int xMidpoint, int yMidpoint);
void writeToImage(outputImageConfig config);

void createOutputTestTrack(NeuralNetwork network, int samplesPerChunk, int frequencyResolution, int chunkBorder);

#endif