#include "Headers/files.h"
#include "Headers/audio.h"

#include <string>
#include <fstream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "Headers/stb_image_write.h"

vector<vector<float>> generateInputs(audioFileConfig config) {
	int endIndex = config.startFileIndex + config.songCount;
	vector<vector<float>> result;

	for (int f = config.startFileIndex; f < endIndex; f++) {
		string fileName = "inputs/" + to_string(f) + ".mp3";
		vector<vector<float>> fullAudioInput = spectrogramOutput(fileName.data(), config.samplesPerChunk, config.samplesPerOverlap, config.frequencyResolution);
		int newFrequencyResolution = fullAudioInput[0].size();

		for (int i = config.chunkBorder; i < fullAudioInput.size() - config.chunkBorder; i++) {
			vector<float> currentInput;

			for (int c = i - config.chunkBorder; c < i + config.chunkBorder; c++) {
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

vector<vector<float>> generateOutputs(audioFileConfig config) {
	int endIndex = config.startFileIndex + config.songCount;
	vector<vector<float>> result;

	for (int f = config.startFileIndex; f < endIndex; f++) {
		string fileName = "outputs/" + to_string(f) + ".mp3";
		vector<vector<float>> fullAudioInput = spectrogramOutput(fileName.data(), config.samplesPerChunk, config.samplesPerOverlap, config.frequencyResolution);
		int newFrequencyResolution = fullAudioInput[0].size();

		for (int i = config.chunkBorder; i < fullAudioInput.size() - config.chunkBorder; i++) {
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

	string fileName = "";
	index = 1;

	while (true) {
		// check if filename exists
		string currentFilename = "config_" + to_string(index) + ".jpg";
		ifstream file(currentFilename.c_str());

		if (!file.good()) {
			fileName = currentFilename;
			break;
		}
		index++;
	}

	stbi_write_jpg(fileName.c_str(), config.errorResolution, config.errorRange, 3, data, config.errorResolution * 3);

	// Write Config Metadata
	ofstream imageFile;
	imageFile.open(fileName.c_str(), ios_base::app); // Append
	imageFile << endl; // New line
	imageFile << "METADATA" << endl;

	// Network config
	imageFile << "Nodes: ";
	for (int i = 0; i < config.network.layerCount; i++) {
		imageFile << to_string(config.network.layerNodes[i].size()) << ", ";
	}

	imageFile << "Biases: ";
	for (int i = 0; i < config.network.layerCount; i++) {
		imageFile << to_string(config.network.layerBiases[i].size()) << ", ";
	}

	imageFile << endl;

	// Training Config
	imageFile << "Epochs: " << to_string(config.trainConfig.epochs) << ", ";
	imageFile << "LR: " << to_string(config.trainConfig.learningRate) << ", ";
	imageFile << "M: " << to_string(config.trainConfig.momentum) << ", ";
	imageFile << "Cycle: " << to_string(config.trainConfig.useCyclicalLearningRateAndMomentum) << ", ";
	imageFile << "Decay: " << to_string(config.trainConfig.useWeightDecay) << ", ";
	imageFile << "Multiplier: " << to_string(config.trainConfig.weightDecayMultiplier) << ", ";
	imageFile << "RPROPd: " << to_string(config.trainConfig.rpropWeightDecreaseMultiplier) << ", ";
	imageFile << "RPROPi: " << to_string(config.trainConfig.rpropWeightIncreaseMultiplier) << ", ";
	imageFile << "Drop: " << to_string(config.trainConfig.useDropout) << ", ";
	imageFile << "Probability: " << to_string(config.trainConfig.nodeBiasDropoutProbability);
	imageFile << endl;

	// Audio Config
	imageFile << "Chunk: " << to_string(config.audioConfig.samplesPerChunk) << ", ";
	imageFile << "Overlap: " << to_string(config.audioConfig.samplesPerOverlap) << ", ";
	imageFile << "Res: " << to_string(config.audioConfig.frequencyResolution) << ", ";
	imageFile << "Border: " << to_string(config.audioConfig.chunkBorder) << ", ";
	imageFile << "Count: " << to_string(config.audioConfig.songCount);
	imageFile << endl;

	// Image Config
	imageFile << "ScaleLow: " << to_string(minScale) << ", ";
	imageFile << "ScaleHigh: " << to_string(maxScale);
}

void createOutputTestTrack(NeuralNetwork network, audioFileConfig config) {
	config.startFileIndex = 1;
	config.songCount = 1;

	vector<vector<float>> testTrackSpectrogram = generateInputs(config); // First track only, for testing
	vector<vector<float>> predictedTrackSpectrogram;

	int chunkCount = testTrackSpectrogram.size();
	int networkLayerCount = network.layerNodes.size();

	for (int i = 0; i < chunkCount; i++) {
		vector<float> currentChunkPrection = network.predict(testTrackSpectrogram[i]);
		predictedTrackSpectrogram.push_back(currentChunkPrection);
	}

	vector<int16_t> testTrackOutputSamples = vocalSamples("inputs/1.mp3", config.samplesPerChunk, config.samplesPerChunk, predictedTrackSpectrogram);
	writeToWAV("testTrackOutput.wav", testTrackOutputSamples);
}
