#include "Headers/files.h"
#include "Headers/audio.h"

#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "Headers/stb_image_write.h"

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

void writeToImage(outputImageConfig config) {
	// ErrorResolution must be a factor of len(errors)
	int errorCount = config.errors.size();

	// Make Boundaries
	float maxScale = 0.0f;
	float minScale = numeric_limits<float>().max();

	if (!config.useFixedScale) {
		for (int i = 0; i < errorCount; i++) {
			maxScale = max(maxScale, config.errors[i]);
			minScale = min(minScale, config.errors[i]);
		}
	}
	if (config.useFixedScale) {
		maxScale = config.fixedMax;
		minScale = config.fixedMin;
	}

	// Normalise Errors
	for (int i = 0; i < errorCount; i++) {
		config.errors[i] = (config.errors[i] / maxScale) * (config.errorRange - 1);
	}

	// Average Out Error Pixels
	int vectorsPerPixel = errorCount / config.errorResolution;
	vector<vector<float>> pixelValues;
	
	// Add Errors
	for (int i = 0; i < errorCount; i += vectorsPerPixel) {
		vector<float> current(config.errorRange);

		for (int j = 0; j < vectorsPerPixel; j++) {
			int index = int(config.errors[i + j]);
			index = min(index, config.errorRange - 1);
			index = max(index, 0);

			current[index] += 1;
		}

		for (int j = 0; j < vectorsPerPixel; j++) {
			current[j] = current[j] / vectorsPerPixel;
		}

		pixelValues.push_back(current);
	}

	// Add Scale Text
	string maxErrorString = to_string(int(maxScale));
	int numberCountInMaxError = to_string(int(maxScale)).length();

	for (int i = 0; i < numberCountInMaxError; i++) {
		int currentCharacterNumber = maxErrorString.at(i) - '0';

		int xMidpoint = 5 * i + i + 3;
		int yMidpoint = config.errorRange - 4;

		pixelValues = addCharacterToImage(pixelValues, currentCharacterNumber, xMidpoint, yMidpoint);
	}

	string minErrorString = to_string(int(minScale));
	for (int i = 0; i < minErrorString.length(); i++) {
		int currentCharacterNumber = minErrorString.at(i) - '0';

		int xMidpoint = 5 * i + i + 3;
		pixelValues = addCharacterToImage(pixelValues, currentCharacterNumber, xMidpoint, 3);
	}

	// Write To Image
	unsigned char* data = new unsigned char[config.errorResolution * config.errorRange * 3];
	int index = 0;

	for (int y = config.errorRange - 1; y >= 0; y -= 1) {
		for (int x = 0; x < config.errorResolution; x++) {
			data[index++] = (unsigned char)(255.0 * pixelValues[x][y]);
			data[index++] = (unsigned char)(255.0 * pixelValues[x][y]); // Light blue multipliers
			data[index++] = (unsigned char)(255.0 * pixelValues[x][y]);
		}
	}

	string fileName = "NETWORKNODES(BIAS),";

	for (int i = 0; i < config.network.layerCount; i++) {
		fileName += to_string(config.network.layerNodes[i].size());
		fileName += "(" + to_string(config.network.layerBiases[i].size()) + ")";

		if (i < config.network.layerCount - 1) {
			fileName += ",";
		}
	}

	fileName += ".jpg";

	stbi_write_jpg(fileName.c_str(), config.errorResolution, config.errorRange, 3, data, config.errorResolution * 3);
}

void createOutputTestTrack(NeuralNetwork network, int samplesPerChunk, int frequencyResolution, int chunkBorder) {
	vector<vector<float>> testTrackSpectrogram = generateInputs(samplesPerChunk, samplesPerChunk, frequencyResolution, chunkBorder, 1, 2); // First track only, for testing
	vector<vector<float>> predictedTrackSpectrogram;

	int chunkCount = testTrackSpectrogram.size();
	int networkLayerCount = network.layerNodes.size();

	for (int i = 0; i < chunkCount; i++) {
		vector<float> currentChunkPrection = network.predict(testTrackSpectrogram[i]);
		predictedTrackSpectrogram.push_back(currentChunkPrection);
	}

	vector<int16_t> testTrackOutputSamples = vocalSamples("inputs/1.mp3", samplesPerChunk, samplesPerChunk, predictedTrackSpectrogram);
	writeToWAV("testTrackOutput.wav", testTrackOutputSamples);
}
