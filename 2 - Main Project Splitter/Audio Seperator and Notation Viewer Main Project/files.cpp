#include "Headers/files.h"
#include "Headers/audio.h"

#include <string>
#include <fstream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "Headers/stb_image_write.h"

// General
string vectorToString(vector<float> used) {
	string result = "(";

	int size = used.size();
	for (int i = 0; i < size; i++) {
		result += to_string(used[i]) + ", ";
	}

	result.pop_back();
	result.pop_back();

	return result + ")";
}
void writeToFile(const char* fileName, vector<string> lines) {
	ofstream currentFile;
	currentFile.open(fileName);

	if (!currentFile) {
		cout << "File could not be opened: " << fileName << endl;
		return;
	}

	int vectorSize = lines.size();
	for (int i = 0; i < vectorSize; i++) {
		currentFile << lines[i] << endl;
	}
	currentFile.close();
}
vector<string> readFile(const char* fileName) {
	vector<string> result;

	ifstream newFile(fileName);
	string currentLine;

	if (!newFile) {
		cout << "File could not be opened: " << fileName << endl;
	}

	while (getline(newFile, currentLine)) {
		result.push_back(currentLine);
	}

	return result;
}
vector<string> splitStringByCharacter(string used, char splitter) {
	vector<string> result;
	stringstream stringStream(used);

	while (stringStream.good()) {
		string substring;
		getline(stringStream, substring, splitter);
		result.push_back(substring);
	}


	return result;
}

// Audio
vector<vector<float>> generateInputs(audioFileConfig config) {
	int endIndex = config.startFileIndex + config.songCount;
	vector<vector<float>> result;

	for (int f = config.startFileIndex; f < endIndex; f++) {
		// Load File Spectrogram From Integer File
		string fileName = "inputs/" + to_string(f) + ".mp3";
		pair<vector<vector<float>>, float> spectrogram = spectrogramOutput(fileName.data(), config);
		vector<vector<float>> fullAudioInput = spectrogram.first;

		// Create Chunk Border to Give Audio 'Context'
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
		// Load File Spectrogram From Integer File
		string fileName = "outputs/" + to_string(f) + ".mp3";
		pair<vector<vector<float>>, float> spectrogram = spectrogramOutput(fileName.data(), config);
		vector<vector<float>> fullAudioInput = spectrogram.first;

		// Create Chunk Border to Give Audio 'Context'
		int newFrequencyResolution = fullAudioInput[0].size();

		for (int i = config.chunkBorder; i < fullAudioInput.size() - config.chunkBorder; i++) {
			vector<float> currentInput;
			for (int f = 0; f < newFrequencyResolution; f++) {
				float value = fullAudioInput[i][f];
				
				// Remove NaN values, very very rare bug in visual studio
				if (isnan(value)) {
					value = 0.0f;
				}

				// Add Optional Binary Mask
				if (config.useOutputBinaryMask) {
					if (value > config.binaryMaskThreshold) {
						value = 1.0f;
					}
					else {
						value = 0.0f;
					}
				}
				
				if (config.useNoisePrediction) {
					value = 1.0f - value; // Noise prediction (invert the amplitude)
				}

				currentInput.push_back(value);
			}
			result.push_back(currentInput);
		}
	}

	return result;
}
void createOutputTestTrack(NeuralNetwork network, audioFileConfig config) {
	config.startFileIndex = 1;
	config.songCount = 1;

	// Get Input Spectrogram (from first file only)
	vector<vector<float>> testTrackSpectrogram = generateInputs(config); // First track only, for testing
	vector<vector<float>> predictedTrackSpectrogram;

	int chunkCount = testTrackSpectrogram.size();
	int networkLayerCount = network.layerNodes.size();

	// Get Network Predictions and Add to Output Track
	for (int i = 0; i < chunkCount; i++) {
		vector<float> currentChunkPrection = network.predict(testTrackSpectrogram[i]);
		predictedTrackSpectrogram.push_back(currentChunkPrection);
	}

	// Get Samples and Write To Track
	vector<int16_t> testTrackOutputSamples = vocalSamples("inputs/1.mp3", predictedTrackSpectrogram, config);
	writeToWAV("testTrackOutput.wav", testTrackOutputSamples);
}

pair<vector<vector<float>>, vector<vector<float>>> generateAllSongDataSet(audioFileConfig config, int chunksPerSong) {
	vector<vector<float>> resultantInputs;
	vector<vector<float>> resultantOutputs;

	config.startFileIndex = 0;
	config.songCount = 1;

	cout << "Loaded Song: ";
	for (int i = 0; i < 100; i++) { // 100 song count
		config.startFileIndex = config.startFileIndex + 1; // Go onto next file

		vector<vector<float>> songInputs = generateInputs(config);
		vector<vector<float>> songOutputs = generateOutputs(config);
		
		// Make sure randomly taken samples are regularly distributed across the song
		int miniBatchSize = songInputs.size() / chunksPerSong;

		for (int j = 0; j < chunksPerSong; j++) {
			int currentIndex = (i * miniBatchSize) + (rand() % miniBatchSize);

			if (songOutputs[currentIndex].size() == 0) {
				continue;
			}

			resultantInputs.push_back(songInputs[currentIndex]);
			resultantOutputs.push_back(songOutputs[currentIndex]);
		}
		cout << i + 1 << ", ";
	}
	cout << "Dataset Size: " << resultantInputs.size() << endl;
	
	pair<vector<vector<float>>, vector<vector<float>>> resultantDataset = make_pair(resultantInputs, resultantOutputs);
	return resultantDataset;
}

// Image
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

	// Normalise Errors (0 - 1 scale)
	for (int i = 0; i < errorCount; i++) {
		config.errors[i] = (config.errors[i] / maxScale) * (config.errorRange - 1);
	}

	// Average Out Error Pixels
	int vectorsPerPixel = errorCount / config.errorResolution;
	vector<vector<float>> pixelValues;
	
	// Add Errors
	for (int i = 0; i < errorCount; i += vectorsPerPixel) {
		// Current Y Pixels (errorCount ammount)
		vector<float> current(config.errorRange);

		// Determine Where Errors Sit on the Y Axis and make Most Common Error Area Brightest
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

	// Create Filename According to What Output Image Configs Already Exist
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
	string trainType = "";
	if (config.trainConfig.trainType == STOCHASTIC_GRADIENT_DESCENT) {
		trainType = "STOCHASTIC_GRADIENT_DESCENT";
	}
	if (config.trainConfig.trainType == GRADIENT_DESCENT) {
		trainType = "GRADIENT_DESCENT";
	}
	if (config.trainConfig.trainType == RESISTANT_PROPAGATION) {
		trainType = "RESISTANT_PROPAGATION";
	}
	if (config.trainConfig.trainType == RANDOM_METHOD) {
		trainType = "RANDOM_METHOD";
	}
	if (config.trainConfig.trainType == LEVENBERG_MARQUARDT) {
		trainType = "LEVENBERG_MARQUARDT";
	}


	imageFile << "Train Method: " << trainType << ", ";
	imageFile << "Epochs: " << to_string(config.trainConfig.epochs) << ", ";
	imageFile << "LR: " << to_string(config.trainConfig.learningRate) << ", ";
	imageFile << "M: " << to_string(config.trainConfig.momentum) << ", ";

	string learningRateType = "";
	if (config.trainConfig.gradientDescent.learningRateType == FIXED_LEARNING_RATE) {
		learningRateType = "FIXED_LR";
	}
	if (config.trainConfig.gradientDescent.learningRateType == CYCLICAL_LEARNING_RATE) {
		learningRateType = "CYCLICAL_LR";
	}
	if (config.trainConfig.gradientDescent.learningRateType == ADAM_LEARNING_RATE) {
		learningRateType = "ADAM_LR";
	}
	imageFile << "Cycle: " << learningRateType << ", ";

	imageFile << "Decay: " << to_string(config.trainConfig.useWeightDecay) << ", ";
	imageFile << "Multiplier: " << to_string(config.trainConfig.weightDecayMultiplier) << ", ";
	imageFile << "RPROPd: " << to_string(config.trainConfig.resistantPropagation.rpropWeightDecreaseMultiplier) << ", ";
	imageFile << "RPROPi: " << to_string(config.trainConfig.resistantPropagation.rpropWeightIncreaseMultiplier) << ", ";
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