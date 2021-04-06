#include "Headers/NeuralNetwork.h"
#include "Headers/audio.h"

#include <iostream>
using namespace std;

#include <chrono>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "Headers/stb_image_write.h"

void writeToImage(vector<float> errors, int errorResolution, int errorRange) {
	// ErrorResolution must be a factor of len(errors)
	// Normalise Errors
	float maxError = 0.0f;
	int errorCount = errors.size();

	for (int i = 0; i < errorCount; i++) {
		maxError = max(maxError, errors[i]);
	}

	for (int i = 0; i < errorCount; i++) {
		errors[i] = (errors[i] / maxError) * (errorRange - 1);
	}

	// Average Out Error Pixels
	int vectorsPerPixel = errorCount / errorResolution;
	vector<vector<float>> pixelValues;

	for (int i = 0; i < errorCount; i += vectorsPerPixel) {
		vector<float> current(errorRange);

		for (int j = 0; j < vectorsPerPixel; j++) {
			current[int(errors[i + j])] += 1;
		}

		for (int j = 0; j < vectorsPerPixel; j++) {
			current[j] = current[j] / vectorsPerPixel;
		}

		pixelValues.push_back(current);
	}

	// Write To Image
	unsigned char* data = new unsigned char[errorResolution * errorRange * 3];
	int index = 0;

	for (int y = errorRange - 1; y >= 0; y -= 1) {
		for (int x = 0; x < errorResolution; x++) {
			data[index++] = (unsigned char)(255.0 * pixelValues[x][y]);
			data[index++] = (unsigned char)(255.0 * pixelValues[x][y]);
			data[index++] = (unsigned char)(255.0 * pixelValues[x][y]);
		}
	}

	stbi_write_jpg("errorOutput.jpg", errorResolution, errorRange, 3, data, errorResolution * 3);
}

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
	// Initial Variables
	int samplesPerChunk = 2048; // I think this should be a power of 2
	int samplesPerOverlap = samplesPerChunk; // no overlap

	int frequencyResolution = 128; // Each float represents (sampleRate / frequencyResolution) frequencies
	int chunkBorder = 4; // How many chunks are added to each side of the input chunk, giving audio "context"
	
	int epochs = 1000;
	float lr = 0.1;
	float momentum = 0.25f;

	int songsPerTrain = 5;

	// Train Network
	
	/*
	vector<vector<float>> inputSet = generateInputs(samplesPerChunk, samplesPerOverlap, frequencyResolution, chunkBorder, 1, songsPerTrain + 1);
	vector<vector<float>> outputSet = generateOutputs(samplesPerChunk, samplesPerOverlap, frequencyResolution, chunkBorder, 1, songsPerTrain + 1);

	int inputSize = inputSet[0].size();
	int outputSize = outputSet[0].size();

	vector<int> layers = { inputSize, outputSize, outputSize, outputSize };
	vector<int> biases = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  };

	NeuralNetwork network = NeuralNetwork(layers, biases, "sigmoid");
	network.loadWeightsFromFile("outputWeights/");

	vector<float> trainingErrors = network.train(inputSet, outputSet, epochs, lr, momentum);

	for (int i = songsPerTrain + 1; i < 101; i += songsPerTrain) {
		inputSet = generateInputs(samplesPerChunk, samplesPerOverlap, frequencyResolution, chunkBorder, i, i + songsPerTrain);
		outputSet = generateOutputs(samplesPerChunk, samplesPerOverlap, frequencyResolution, chunkBorder, i, i + songsPerTrain);

		vector<float> currentTrainingErrors = network.train(inputSet, outputSet, epochs, lr, momentum);
		trainingErrors.insert(trainingErrors.end(), currentTrainingErrors.begin(), currentTrainingErrors.end());
	}
	
	network.saveWeightsToFile("outputWeights/");
	*/

	vector<vector<float>> inputSet = generateInputs(samplesPerChunk, samplesPerOverlap, frequencyResolution, chunkBorder, 1, 2);
	vector<vector<float>> outputSet = generateOutputs(samplesPerChunk, samplesPerOverlap, frequencyResolution, chunkBorder, 1, 2);

	int inputSize = inputSet[0].size();
	int outputSize = outputSet[0].size();

	vector<int> layers = { inputSize, outputSize * 4, outputSize * 6, outputSize * 2, outputSize, outputSize };
	vector<int> biases = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, };

	NeuralNetwork network = NeuralNetwork(layers, biases, "tanh");

	vector<float> trainingErrors = network.train(inputSet, outputSet, epochs, lr, momentum);
	writeToImage(trainingErrors, 1000, 512);
	//network.saveWeightsToFile("outputWeights/");

	// Test with first test songs
	vector<vector<float>> testTrackSpectrogram = generateInputs(samplesPerChunk, samplesPerChunk, frequencyResolution, chunkBorder, 1, 2); // First track only, for testing
	vector<vector<float>> predictedTrackSpectrogram;

	int chunkCount = testTrackSpectrogram.size();
	int networkLayerCount = layers.size();
	
	for (int i = 0; i < chunkCount; i++) {
		vector<float> currentChunkPrection = network.predict(testTrackSpectrogram[i]);
		predictedTrackSpectrogram.push_back(currentChunkPrection);
	}

	vector<int16_t> testTrackOutputSamples = vocalSamples("inputs/1.mp3", samplesPerChunk, samplesPerChunk, predictedTrackSpectrogram);
	writeToWAV("testTrackOutput.wav", testTrackOutputSamples);

	system("pause");
	return 0;
}