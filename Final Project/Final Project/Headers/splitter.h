#ifndef SPLITTER_H
#define SPLITTER_H

#define STEM_VOCAL 1
#define STEM_BASS 2
#define STEM_DRUMS 3

#define STEMS_VOCALS_BACKING 1
#define STEMS_ALL 2

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
	
	int predictionsDone = 0;
	int predictionsRequired = 0;
	void predictTrackStemToFile(const char* inputFilename, int STEM, const char* outputFilename);

	static vector<vector<float>> flipOutputVector(vector<vector<float>> input);
	static vector<vector<float>> addOutputVectors(vector<vector<float>> inputOne, vector<vector<float>> inputTwo);

	void splitStems(int STEMS_CHOICE, const char* inputFilename, string outputDirectory);

	vector<vector<int16_t>> outputSamples;
};

#endif // !SPLITTER_H
