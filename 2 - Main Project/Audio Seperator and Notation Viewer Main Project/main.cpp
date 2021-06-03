#include "Headers/NeuralNetwork.h"

#include <iostream>
using namespace std;

#include "Headers/files.h"
#include "Headers/audio.h"

int main() {
	// Initial Variables
	int samplesPerChunk = 2048; // I think this should be a power of 2
	int samplesPerOverlap = samplesPerChunk; // no overlap
	int frequencyResolution = 128; // Each float represents (sampleRate / frequencyResolution) frequencies
	int chunkBorder = 4; // How many chunks are added to each side of the input chunk, giving audio "context"
	int songsPerTrain = 1;

	// Train Network - One Song Training
	vector<vector<float>> inputSet = generateInputs(samplesPerChunk, samplesPerOverlap, frequencyResolution, chunkBorder, 1, songsPerTrain + 1);
	vector<vector<float>> outputSet = generateOutputs(samplesPerChunk, samplesPerOverlap, frequencyResolution, chunkBorder, 1, songsPerTrain + 1);

	int inputSize = inputSet[0].size();
	int outputSize = outputSet[0].size();

	vector<int> layers = { inputSize, outputSize, };
	vector<int> biases = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	NeuralNetwork network = NeuralNetwork(layers, biases, "tanh");
	//network.loadWeightsFromFile("outputWeights/");
	
	standardTrainConfig trainingConfig = {
		inputSet,
		outputSet,

		1000, // Epochs

		1.0f, // LR
		0.25f, // Momentum

		true, // Use Cyclical Learning Rate?

		false, // Use Weight Decay ?
		0.99f, // Weight Decay Multiplier

		0.5f, // RPROP Weight Decrease
		1.2f, // RPROP Weight Increase

		false, // Use Dropout ?
		4, // 1 in "x" Random Nodes / Biases dropped
	};
	vector<float> trainingErrors = network.train(trainingConfig);

	outputImageConfig imageConfig{
		trainingErrors,

		1000, // error res
		512, // error range

		network,

		false, // Use fixed scale?
		500.0f, // max scale on fixed scale
		0.0f, // min scale on fixed scale
	};
	writeToImage(imageConfig);

	//network.saveWeightsToFile("outputWeights/");

	// Test with first test songs
	createOutputTestTrack(network, samplesPerChunk, frequencyResolution, chunkBorder);

	system("pause");
	return 0;
}