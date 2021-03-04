#include "Headers/NeuralNetwork.h"
#include "Headers/audio.h"

#include <iostream>
using namespace std;

vector<vector<float>> generateInputs(int samplesPerChunk, int frequencyResolution, int chunksPerInputHalf, int zeroRange) {
	vector<vector<float>> result;

	for (int f = 7; f < 10; f++) {
		string fileName = "inputs/" + to_string(f) + ".mp3";
		vector<vector<float>> fullAudioInput = spectrogramOutput(fileName.data(), samplesPerChunk, frequencyResolution, zeroRange, false);

		for (int i = chunksPerInputHalf; i < fullAudioInput.size() - chunksPerInputHalf; i++) {
			vector<float> currentInput;

			for (int c = i - chunksPerInputHalf; c < i + chunksPerInputHalf; c++) {
				for (int f = 0; f < frequencyResolution; f++) {
					currentInput.push_back(fullAudioInput[c][f]);
				}
			}

			result.push_back(currentInput);
		}
	}

	return result;
}

vector<vector<float>> generateOutputs(int samplesPerChunk, int frequencyResolution, int chunksPerInputHalf, int zeroRange) {
	vector<vector<float>> result;

	for (int f = 5; f < 10; f++) {
		string fileName = "outputs/" + to_string(f) + ".mp3";
		vector<vector<float>> fullAudioInput = spectrogramOutput(fileName.data(), samplesPerChunk, frequencyResolution, zeroRange, true);

		for (int i = chunksPerInputHalf; i < fullAudioInput.size() - chunksPerInputHalf; i++) {
			vector<float> currentInput;
			for (int f = 0; f < frequencyResolution; f++) {
				if (fullAudioInput[i][f] > zeroRange) {
					//fullAudioInput[i][f] = 1.0f;
				}
				if (fullAudioInput[i][f] <= zeroRange) {
					//fullAudioInput[i][f] = 0.0f;
				}

				currentInput.push_back(fullAudioInput[i][f]);
			}
			result.push_back(currentInput);
		}
	}

	return result;
}

int main() {
	int samplesPerChunk = 1000;
	int frequencyResolution = 25;
	int zeroRange = 500;
	int chunkBorder = 20;

	vector<vector<float>> inputSet = generateInputs(samplesPerChunk, frequencyResolution, chunkBorder, zeroRange);
	vector<vector<float>> outputSet = generateOutputs(samplesPerChunk, frequencyResolution, chunkBorder, zeroRange);

	int inputSize = inputSet[0].size();
	int outputSize = outputSet[0].size();

	vector<int> layers = { inputSize, inputSize * 2, outputSize };
	vector<int> biases = { 1, 1, 1,1 ,1, 1,1 ,1  };

	NeuralNetwork network = NeuralNetwork(layers, biases, "sigmoid");
	//network.loadWeightsFromFile("outputWeights/");
	network.train(inputSet, outputSet, 4, 0.25f, 0.0f);
	network.saveWeightsToFile("outputWeights/");

	system("pause");
	return -1;
}