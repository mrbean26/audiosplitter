#include "Headers/NeuralNetwork.h"
#include "Headers/audio.h"

#include <iostream>
using namespace std;

#include <chrono>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "Headers/stb_image_write.h"

vector<vector<float>> addCharacterToImage(vector<vector<float>> data, int character, int xMidpoint, int yMidpoint) {
	vector<vector<float>> characterPixels; // 5 x 5

	if (character == 0) {
		characterPixels = { {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
							{1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
							{1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
							{1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
							{1.0f, 1.0f, 1.0f, 1.0f, 1.0f} };
	}
	if (character == 1) {
		characterPixels = { {0.0f, 0.0f, 1.0f, 0.0f, 0.0f},
							{0.0f, 0.0f, 1.0f, 0.0f, 0.0f},
							{0.0f, 0.0f, 1.0f, 0.0f, 0.0f},
							{0.0f, 0.0f, 1.0f, 0.0f, 0.0f},
							{0.0f, 0.0f, 1.0f, 0.0f, 0.0f} };
	}
	if (character == 2) {
		characterPixels = { {1.0f, 1.0f, 1.0f, 1.0f, 0.0f},
							{0.0f, 0.0f, 1.0f, 0.0f, 0.0f},
							{0.0f, 1.0f, 0.0f, 0.0f, 0.0f},
							{1.0f, 0.0f, 0.0f, 0.0f, 0.0f},
							{1.0f, 1.0f, 1.0f, 1.0f, 1.0f} };
	}
	if (character == 3) {
		characterPixels = { {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
							{0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
							{1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
							{0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
							{1.0f, 1.0f, 1.0f, 1.0f, 1.0f} };
	}
	if (character == 4) {
		characterPixels = { {0.0f, 0.0f, 1.0f, 1.0f, 0.0f},
							{0.0f, 1.0f, 0.0f, 1.0f, 0.0f},
							{1.0f, 0.0f, 0.0f, 1.0f, 0.0f},
							{1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
							{0.0f, 0.0f, 0.0f, 1.0f, 0.0f} };
	}
	if (character == 5) {
		characterPixels = { {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
							{1.0f, 0.0f, 0.0f, 0.0f, 0.0f},
							{1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
							{0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
							{1.0f, 1.0f, 1.0f, 1.0f, 1.0f} };
	}
	if (character == 6) {
		characterPixels = { {1.0f, 0.0f, 0.0f, 0.0f, 0.0f},
							{1.0f, 0.0f, 0.0f, 0.0f, 0.0f},
							{1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
							{1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
							{1.0f, 1.0f, 1.0f, 1.0f, 1.0f} };
	}
	if (character == 7) {
		characterPixels = { {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
							{0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
							{0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
							{0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
							{0.0f, 0.0f, 0.0f, 0.0f, 1.0f} };
	}
	if (character == 8) {
		characterPixels = { {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
							{1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
							{1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
							{1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
							{1.0f, 1.0f, 1.0f, 1.0f, 1.0f} };
	}
	if (character == 9) {
		characterPixels = { {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
							{1.0f, 0.0f, 0.0f, 1.0f, 1.0f},
							{1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
							{0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
							{0.0f, 0.0f, 0.0f, 0.0f, 1.0f} };
	}

	// Write to Image
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			data[i - 2 + xMidpoint][4 - j - 2 + yMidpoint] = characterPixels[j][i];
		}
	}


	return data;
}

void writeToImage(vector<float> errors, int errorResolution, int errorRange, NeuralNetwork network) {
	// ErrorResolution must be a factor of len(errors)
	// Normalise Errors
	float maxError = 0.0f;
	float minError = numeric_limits<float>().max();
	int errorCount = errors.size();

	for (int i = 0; i < errorCount; i++) {
		maxError = max(maxError, errors[i]);
		minError = min(minError, errors[i]);
	}

	for (int i = 0; i < errorCount; i++) {
		errors[i] = (errors[i] / maxError) * (errorRange - 1);
	}


	// Average Out Error Pixels
	int vectorsPerPixel = errorCount / errorResolution;
	vector<vector<float>> pixelValues;

	// Add Scale Background
	int numberCountInMaxError = to_string(int(maxError)).length();
	int scaleXSize = 5 * numberCountInMaxError + numberCountInMaxError;

	for (int i = 0; i < scaleXSize; i++) {
		vector<float> newVector(errorRange);
		pixelValues.push_back(newVector);
	}

	// Add Scale Text
	string maxErrorString = to_string(int(maxError));
	for (int i = 0; i < numberCountInMaxError; i++) {
		int currentCharacterNumber = maxErrorString.at(i) - '0';

		int xMidpoint = 5 * i + i + 3;
		int yMidPoint = errorRange - 4;

		pixelValues = addCharacterToImage(pixelValues, currentCharacterNumber, xMidpoint, yMidPoint);
	}

	string minErrorString = to_string(int(minError));
	for (int i = 0; i < minErrorString.length(); i++) {
		int currentCharacterNumber = minErrorString.at(i) - '0';

		int xMidpoint = 5 * i + i + 3;
		pixelValues = addCharacterToImage(pixelValues, currentCharacterNumber, xMidpoint, 3);
	}

	// Add Errors
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

	string fileName = "NETWORKNODES(BIAS),";

	for (int i = 0; i < network.layerCount; i++) {
		fileName += to_string(network.layerNodes[i].size());
		fileName += "(" + to_string(network.layerBiases[i].size()) + ")";

		if (i < network.layerCount - 1) {
			fileName += ",";
		}
	}

	fileName += ".jpg";

	stbi_write_jpg(fileName.c_str(), errorResolution, errorRange, 3, data, errorResolution * 3);
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
	int songsPerTrain = 1;

	// Train Network - One Song Training
	vector<vector<float>> inputSet = generateInputs(samplesPerChunk, samplesPerOverlap, frequencyResolution, chunkBorder, 1, songsPerTrain + 1);
	vector<vector<float>> outputSet = generateOutputs(samplesPerChunk, samplesPerOverlap, frequencyResolution, chunkBorder, 1, songsPerTrain + 1);

	int inputSize = inputSet[0].size();
	int outputSize = outputSet[0].size();

	vector<int> layers = { inputSize, outputSize, };
	vector<int> biases = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

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

	//vector<float> trainingErrors = network.train(trainingConfig);
	vector<float> trainingErrors = network.train(trainingConfig);

	writeToImage(trainingErrors, 1000, 512, network);

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