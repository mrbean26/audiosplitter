#include "Headers/NeuralNetwork.h"
#include "Headers/audio.h"

#include <iostream>
using namespace std;

#include <chrono>

vector<vector<float>> generateInputs(int samplesPerChunk, int samplesPerOverlap, int frequencyResolution, int chunksPerInputHalf) {
	vector<vector<float>> result;

	for (int f = 1; f < 3; f++) {
		string fileName = "inputs/" + to_string(f) + ".mp3";
		vector<vector<float>> fullAudioInput = spectrogramOutput(fileName.data(), samplesPerChunk, samplesPerOverlap, frequencyResolution);

		for (int i = chunksPerInputHalf; i < fullAudioInput.size() - chunksPerInputHalf; i++) {
			vector<float> currentInput;

			for (int c = i - chunksPerInputHalf; c < i + chunksPerInputHalf; c++) {
				for (int f = 0; f < frequencyResolution / 4; f++) {
					float value = fullAudioInput[i][f];

					// Remove NaN values, very very rare bug in visual studio
					if (isnan(value)) {
						value = 0.0f;
					}

					currentInput.push_back(value);
				}
			}

			result.push_back(currentInput);
		}
	}

	return result;
}

vector<vector<float>> generateOutputs(int samplesPerChunk, int samplesPerOverlap, int frequencyResolution, int chunksPerInputHalf) {
	vector<vector<float>> result;

	for (int f = 1; f < 3; f++) {
		string fileName = "outputs/" + to_string(f) + ".mp3";
		vector<vector<float>> fullAudioInput = spectrogramOutput(fileName.data(), samplesPerChunk, samplesPerOverlap, frequencyResolution);

		for (int i = chunksPerInputHalf; i < fullAudioInput.size() - chunksPerInputHalf; i++) {
			vector<float> currentInput;
			for (int f = 0; f < frequencyResolution / 4; f++) {
				float value = fullAudioInput[i][f];

				// Remove NaN values, very very rare bug in visual studio
				if (isnan(value)) {
					value = 0.0f;
				}

				currentInput.push_back(value);
			}
			result.push_back(currentInput);
		}
	}

	return result;
}

int main() {
	int samplesPerChunk = 8192;
	int samplesPerOverlap = samplesPerChunk; // no overlap

	int frequencyResolution = 64;
	int chunkBorder = 20;

	vector<vector<float>> inputSet = generateInputs(samplesPerChunk, samplesPerOverlap, frequencyResolution, chunkBorder);
	vector<vector<float>> outputSet = generateOutputs(samplesPerChunk, samplesPerOverlap, frequencyResolution, chunkBorder);
	
	int inputSize = inputSet[0].size();
	int outputSize = outputSet[0].size();

	vector<int> layers = { inputSize, inputSize, outputSize, outputSize, outputSize };
	vector<int> biases = { 1, 1, 1, 1, 1, 1, 1  };

	NeuralNetwork network = NeuralNetwork(layers, biases, "tanh");
	//network.loadWeightsFromFile("outputWeights/");
	network.train(inputSet, outputSet, 150, 0.05f, 0.0f);
	//network.saveWeightsToFile("outputWeights/");

	system("pause");
	return -1;
}