#ifndef SPLITTER_H
#define SPLITTER_H

#define STEM_VOCAL 1
#define STEM_BASS 2
#define STEM_DRUMS 3

#include "NeuralNetwork.h"

class splitter {
public:
	splitter(); // default constructer
	splitter(int STEM);

	NeuralNetwork predictionNetwork;
	audioFileConfig audioConfig;

	int currentLoadedStemWeights = -1;
	int outputCount = -1;

	void getNetwork();
	void getAudioConfig();

	NeuralNetwork::standardTrainConfig getTrainConfig();
	void trainNetwork(int STEM, const char* weightOutputDirectory);

	void loadStemWeights(int STEM);
	vector<vector<float>> predictTrack(vector<vector<float>> inputs);
	void predictTrackStemToFile(const char* inputFilename, int STEM, const char* outputFilename);
};

#endif // !SPLITTER_H
