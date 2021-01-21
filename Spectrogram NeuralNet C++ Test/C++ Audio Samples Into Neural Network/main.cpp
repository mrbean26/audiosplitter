#include "Headers/NeuralNetwork.h"
#include "Headers/audio.h"

#include <iostream>
using namespace std;

vector<vector<float>> generateInputs(int samplesPerChunk, int frequencyResolution, int chunksPerInputHalf, int zeroRange) {
	vector<vector<float>> result;
	vector<vector<float>> fullAudioInput = spectrogramOutput("full.mp3", samplesPerChunk, frequencyResolution, zeroRange);
	
	for (int i = chunksPerInputHalf; i < fullAudioInput.size() - chunksPerInputHalf; i++) {
		vector<float> currentInput;
		
		for (int c = i - chunksPerInputHalf; c < i + chunksPerInputHalf; c++) {
			for (int f = 0; f < frequencyResolution; f++) {
				currentInput.push_back(fullAudioInput[c][f]);
			}
		}

		result.push_back(currentInput);
	}

	return result;
}

vector<vector<float>> generateOutputs(int samplesPerChunk, int frequencyResolution, int chunksPerInputHalf, int zeroRange) {
	vector<vector<float>> result;
	vector<vector<float>> fullAudioInput = spectrogramOutput("vocals.mp3", samplesPerChunk, frequencyResolution, zeroRange);

	for (int i = chunksPerInputHalf; i < fullAudioInput.size() - chunksPerInputHalf; i++) {
		vector<float> currentInput;
		for (int f = 0; f < frequencyResolution; f++) {
			currentInput.push_back(fullAudioInput[i][f]);
		}
		result.push_back(currentInput);
	}

	return result;
}

int main() {
	int samplesPerChunk = 1000;
	int frequencyResolution = 280;
	int zeroRange = 10;
	
	vector<vector<float>> inputSet = generateInputs(samplesPerChunk, frequencyResolution, 10, 500);
	vector<vector<float>> outputSet = generateOutputs(samplesPerChunk, frequencyResolution, 10, 500);
	
	int inputSize = inputSet[0].size();
	int outputSize = outputSet[0].size();

	vector<int> layers = { inputSize, (int)(inputSize * 1.25), (int)(inputSize * 1.5), outputSize };
	vector<int> biases = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	NeuralNetwork network = NeuralNetwork(layers, biases, "tanh");
	network.train(inputSet, outputSet, 1000, 0.05f, 0.25f);

	system("pause");
	return -1;
}