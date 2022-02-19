#ifndef FILES_H
#define FILES_H

#include "Headers/NeuralNetwork.h"

#include <vector>
using namespace std;

// General
string vectorToString(vector<float> used);
void writeToFile(const char* fileName, vector<string> lines);
vector<string> readFile(const char* fileName);
vector<string> splitStringByCharacter(string used, char splitter);

// Classes
struct outputImageConfig {
	vector<float> errors;

	int errorResolution;
	int errorRange;

	NeuralNetwork network;
	audioFileConfig audioConfig;
	NeuralNetwork::standardTrainConfig trainConfig;

	bool useFixedScale = true;
	float fixedMax = 5000.0f;
	float fixedMin = 0.0f;
};

// Audio
vector<vector<float>> generateInputs(audioFileConfig config);
vector<vector<float>> generateOutputs(audioFileConfig config);
void createOutputTestTrack(NeuralNetwork network, audioFileConfig config);

pair<vector<vector<float>>, vector<vector<float>>> generateAllSongDataSet(audioFileConfig config, int chunksPerSong);

// Image
vector<vector<float>> addCharacterToImage(vector<vector<float>> data, int character, int xMidpoint, int yMidpoint);
void writeToImage(outputImageConfig config);

#endif