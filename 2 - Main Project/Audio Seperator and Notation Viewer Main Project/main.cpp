#include "Headers/NeuralNetwork.h"
#include "Headers/audio.h"

#include <iostream>
using namespace std;

#include <chrono>

vector<vector<float>> generateInputs(int samplesPerChunk, int samplesPerOverlap, int frequencyResolution, int chunksPerInputHalf, int startFileIndex, int endIndex) {
	vector<vector<float>> result;

	for (int f = startFileIndex; f < endIndex; f++) {
		string fileName = "inputs/" + to_string(f) + ".mp3";
		vector<vector<float>> fullAudioInput = spectrogramOutput(fileName.data(), samplesPerChunk, samplesPerOverlap, frequencyResolution);
		int newFrequencyResolution = fullAudioInput[0].size();

		for (int i = chunksPerInputHalf; i < fullAudioInput.size() - chunksPerInputHalf; i++) {
			vector<float> currentInput;

			for (int c = i - chunksPerInputHalf; c < i + chunksPerInputHalf; c++) {
				for (int f = 0; f < newFrequencyResolution; f++) {
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

vector<vector<float>> generateOutputs(int samplesPerChunk, int samplesPerOverlap, int frequencyResolution, int chunksPerInputHalf, int startFileIndex, int endIndex) {
	vector<vector<float>> result;

	for (int f = startFileIndex; f < endIndex; f++) {
		string fileName = "outputs/" + to_string(f) + ".mp3";
		vector<vector<float>> fullAudioInput = spectrogramOutput(fileName.data(), samplesPerChunk, samplesPerOverlap, frequencyResolution);
		int newFrequencyResolution = fullAudioInput[0].size();

		for (int i = chunksPerInputHalf; i < fullAudioInput.size() - chunksPerInputHalf; i++) {
			vector<float> currentInput;
			for (int f = 0; f < newFrequencyResolution; f++) {
				float value = fullAudioInput[i][f];

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
	int samplesPerChunk = 8192; // I think this should be a power of 2
	int samplesPerOverlap = samplesPerChunk; // no overlap

	int frequencyResolution = 512; // Each float represents (sampleRate / frequencyResolution) frequencies
	int chunkBorder = 10; // How many chunks are added to each side of the input chunk, giving audio "context"
	
	int epochs = 10;
	float lr = 0.05;
	float momentum = 0.0f;

	int songsPerTrain = 5;
	
	vector<vector<float>> inputSet = generateInputs(samplesPerChunk, samplesPerOverlap, frequencyResolution, chunkBorder, 1, songsPerTrain + 1);
	vector<vector<float>> outputSet = generateOutputs(samplesPerChunk, samplesPerOverlap, frequencyResolution, chunkBorder, 1, songsPerTrain + 1);
	
	int inputSize = inputSet[0].size();
	int outputSize = outputSet[0].size();

	vector<int> layers = { inputSize, outputSize, outputSize, outputSize };
	vector<int> biases = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  };

	NeuralNetwork network = NeuralNetwork(layers, biases, "tanh");
	network.loadWeightsFromFile("outputWeights/");
	
	network.train(inputSet, outputSet, epochs, lr, momentum);
	for (int i = songsPerTrain + 1; i < 101; i += songsPerTrain) {
		inputSet = generateInputs(samplesPerChunk, samplesPerOverlap, frequencyResolution, chunkBorder, i, i + songsPerTrain);
		outputSet = generateOutputs(samplesPerChunk, samplesPerOverlap, frequencyResolution, chunkBorder, i, i + songsPerTrain);
		network.train(inputSet, outputSet, epochs, lr, momentum);
	}
	
	network.saveWeightsToFile("outputWeights/");

	system("pause");
	return 0;
}