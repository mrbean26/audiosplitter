#include "Headers/files.h"
#include "Headers/audio.h"

#include <string>
#include <fstream>
#include <future>

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

	// Write vector to file line by line
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

	// read file in line by line
	while (getline(newFile, currentLine)) {
		result.push_back(currentLine);
	}

	return result;
}
vector<string> splitStringByCharacter(string used, char splitter) {
	vector<string> result;
	stringstream stringStream(used);

	// traverse string chunks between each character
	while (stringStream.good()) {
		string substring;
		getline(stringStream, substring, splitter);
		result.push_back(substring);
	}

	return result;
}
void outputDataset(vector<vector<float>> data) {
	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < data[i].size(); j++) {
			cout << data[i][j] << ",";
		}
		cout << endl;
		system("pause");
	}
}

// Audio
vector<vector<float>> generateSingleTrackInput(audioFileConfig config, string fileName) {
	vector<vector<float>> result;
	pair<vector<vector<float>>, float> spectrogram = spectrogramOutput(fileName.data(), config);
	vector<vector<float>> fullAudioInput = spectrogram.first;

	// Create Chunk Border to Give Audio 'Context'
	int newFrequencyResolution = fullAudioInput[0].size();
	int chunkStep = 1;
	if (config.skipOverlapChunks) {
		chunkStep = config.samplesPerChunk / config.samplesPerOverlap;
	}

	for (int i = config.chunkBorder; i < fullAudioInput.size() - config.chunkBorder; i += chunkStep) {
		vector<float> currentInput;

		for (int c = i - config.chunkBorder; c < i + config.chunkBorder + 1; c++) {
			for (int f = 0; f < newFrequencyResolution; f++) {
				float value = fullAudioInput[c][f];

				// Remove NaN values, very very rare bug in visual studio
				if (isnan(value)) {
					value = 0.0f;
				}

				currentInput.push_back(value);
			}
		}

		result.push_back(currentInput);
	}

	return result;
}
vector<vector<float>> generateInputs(audioFileConfig config) {
	int endIndex = config.startFileIndex + config.songCount;
	vector<vector<float>> result;
	
	for (int f = config.startFileIndex; f < endIndex; f++) {
		// Load File Spectrogram From Integer File
		string fileName = "inputs/" + to_string(f) + ".mp3";

		vector<vector<float>> currentTrackInputs = generateSingleTrackInput(config, fileName);
		result.insert(result.end(), currentTrackInputs.begin(), currentTrackInputs.end());
	}

	return result;
}
vector<vector<float>> generateOutputs(audioFileConfig config) {
	int endIndex = config.startFileIndex + config.songCount;
	vector<vector<float>> result;

	int chunkStep = 1;
	if (config.skipOverlapChunks) {
		chunkStep = config.samplesPerChunk / config.samplesPerOverlap;
	}

	for (int f = config.startFileIndex; f < endIndex; f++) {
		// Load File Spectrogram From Integer File
		string fileName = "outputs/" + to_string(f) + ".mp3";
		pair<vector<vector<float>>, float> spectrogram = spectrogramOutput(fileName.data(), config);
		vector<vector<float>> fullAudioInput = spectrogram.first;

		// Create Chunk Border to Give Audio 'Context'
		int newFrequencyResolution = fullAudioInput[0].size();

		for (int i = config.chunkBorder; i < fullAudioInput.size() - config.chunkBorder; i += chunkStep) {
			vector<float> currentInput;
			int count = 0;

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
						count = count + 1;
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

			if (config.useSingleOutputValue) {
				int requiredBands = config.singleOutputChunkThreshold * newFrequencyResolution;

				if (count >= requiredBands) {
					currentInput = { 1.0f };
				}
				else {
					currentInput = { 0.0f };
				}
			}

			result.push_back(currentInput);
		}
	}

	return result;
}

void testNetworkInputsToImage(audioFileConfig audioConfig) {
	audioConfig.startFileIndex = 1;
	audioConfig.songCount = 1;

	vector<vector<float>> fullInputs = generateInputs(audioConfig);
	vector<float> input = fullInputs[0];

	vector<vector<float>> inputChunks;
	int inputSize = input.size();

	int totalChunkCount = 2 * audioConfig.chunkBorder + 1;
	int floatsPerChunk = inputSize / totalChunkCount;

	for (int i = 0; i < inputSize; i += floatsPerChunk) {
		vector<float> currentChunk;

		for (int j = 0; j < floatsPerChunk; j++) {
			currentChunk.push_back(input[i + j]);
		}
		inputChunks.push_back(currentChunk);
	}

	writeSpectrogramToImage(inputChunks, "_Testing/testInputChunks.png");
}
void testNetworkOutputsToImage(audioFileConfig audioConfig) {
	audioConfig.startFileIndex = 1;
	audioConfig.songCount = 1;

	vector<vector<float>> fullOutputs = generateOutputs(audioConfig);
	writeSpectrogramToImage(fullOutputs, "_Testing/testOutputChunks.png");
}
void outputInputVector(vector<float> inputVector, audioFileConfig audioConfig) {
	int inputSize = inputVector.size();

	int totalChunkCount = 2 * audioConfig.chunkBorder + 1;
	int floatsPerChunk = inputSize / totalChunkCount;

	for (int i = 0; i < inputSize; i += floatsPerChunk) {
		vector<float> currentChunk;

		for (int j = 0; j < floatsPerChunk; j++) {
			currentChunk.push_back(inputVector[i + j]);
		}

		outputVector(currentChunk);
	}
}
void outputVector(vector<float> vector) {
	int size = vector.size();

	for (int i = 0; i < size; i++) {
		cout << vector[i];

		if (i != size - 1) {
			cout << ", ";
		}
	}

	cout << "." << endl;
}
void inputTrackSpectrogramToImage(audioFileConfig audioConfig) {
	audioConfig.songCount = 1;
	audioConfig.startFileIndex = 1;

	audioConfig.samplesPerOverlap = audioConfig.samplesPerChunk;
	audioConfig.chunkBorder = 0;

	vector<vector<float>> inputTrackSpectrogram = generateInputs(audioConfig);
	writeSpectrogramToImage(inputTrackSpectrogram, "_Testing/inputSpectrogram.png");
}

vector<float> removePredictionNoise(vector<float> networkPrediction, int chunkSize, int chunkCount) {
	vector<float> result;
	int predictionCount = networkPrediction.size();

	int zeroCount = 0; int oneCount = 0;
	for (int i = 0; i < predictionCount; i++) {
		if (networkPrediction[i] < 0.5f) {
			zeroCount = zeroCount + 1;
		}
		else {
			oneCount = oneCount + 1;
		}

		if ((i + 1) % chunkSize == 0) {
			float value = 0.0f;
			if (oneCount >= chunkCount) {
				value = 1.0f;
			}

			for (int k = 0; k < chunkSize; k++) {
				result.push_back(value);
			}

			zeroCount = 0;
			oneCount = 0;
		}
	}

	return result;
}
vector<vector<float>> singleValueToChunks(vector<float> predictions, int size) {
	vector<float> oneChunk(size, 1.0f);
	vector<float> zeroChunk(size, 0.0f);
	
	int chunkCount = predictions.size();
	vector<vector<float>> result;

	for (int i = 0; i < chunkCount; i++) {
		if (predictions[i] == 1.0f) {
			result.push_back(oneChunk);
		}
		else {
			result.push_back(zeroChunk);
		}
	}

	return result;
}

void createOutputTestTrack(NeuralNetwork network, audioFileConfig config, string trackName) {
	// Get Input Spectrogram (from first file only)
	vector<vector<float>> testTrackSpectrogram = generateSingleTrackInput(config, trackName); // First track only, for testing
	vector<vector<float>> predictedTrackSpectrogram;

	int chunkCount = testTrackSpectrogram.size();
	int indexJump = config.samplesPerChunk / config.samplesPerOverlap; // to support overlap output, skip "duplicate" output chunks

	int networkLayerCount = network.layerNodes.size();
	vector<vector<float>> networkPredictions;
	vector<float> singleValuePredictions;

	// Get Network Predictions and Add to Output Track
	for (int i = 0; i < chunkCount; i += indexJump) {
		vector<float> currentChunkPrection = network.predict(testTrackSpectrogram[i]);
		
		// Use step function if binary mask has been used
		if (config.useOutputBinaryMask) {
			for (int j = 0; j < currentChunkPrection.size(); j++) {
				if (currentChunkPrection[j] > 0.85f) {
					currentChunkPrection[j] = 1.0f;
				}
				else {
					currentChunkPrection[j] = 0.0f;
				}
			}
		}
		
		networkPredictions.push_back(currentChunkPrection);

		if (config.useSingleOutputValue) {
			singleValuePredictions.push_back(currentChunkPrection[0]);
		}

		predictedTrackSpectrogram.push_back(currentChunkPrection);
	}





	int averageNumber = 0;
	for (int i = 0; i < predictedTrackSpectrogram.size(); i++) {
		int currentCount = 0;

		for (int j = 0; j < predictedTrackSpectrogram[i].size(); j++) {
			if (predictedTrackSpectrogram[i][j] == 1.0f) {
				currentCount += 1;
			}
		}

		averageNumber += currentCount;
	}
	averageNumber = averageNumber / predictedTrackSpectrogram.size();


	vector<float> singleChunks;
	for (int i = 0; i < predictedTrackSpectrogram.size(); i++) {
		int currentCount = 0;

		for (int j = 0; j < predictedTrackSpectrogram[i].size(); j++) {
			if (predictedTrackSpectrogram[i][j] == 1.0f) {
				currentCount += 1;
			}
		}

		if (currentCount > averageNumber * 0.1) {
			singleChunks.push_back(1.0f);
		}
		else {
			singleChunks.push_back(0.0f);
		}
	}


	singleChunks = removePredictionNoise(singleChunks, config.noiseReductionChunkSize, config.noiseReductionRequiredChunks);


	vector<vector<float>> finalOuts;
	for (int i = 0; i < predictedTrackSpectrogram.size(); i++) {
		if (singleChunks[i] == 0.0f) {
			vector<float> newvec(predictedTrackSpectrogram[i].size(), 0.0f);
			finalOuts.push_back(newvec);
		}
		if (singleChunks[i] == 1.0f) {
			finalOuts.push_back(predictedTrackSpectrogram[i]);
		}
	}


	predictedTrackSpectrogram = finalOuts;










	if (config.useSingleOutputValue) {
		singleValuePredictions = removePredictionNoise(singleValuePredictions, config.noiseReductionChunkSize, config.noiseReductionRequiredChunks);
		vector<vector<float>> v;

		for (int i = 0; i < singleValuePredictions.size(); i++) {
			v.push_back({ singleValuePredictions[i] });
		}
		writeSpectrogramToImage(v, "_Testing/Predictions/vocalPredictionCorrectedSpectrogram.png");

		predictedTrackSpectrogram = singleValueToChunks(singleValuePredictions, config.frequencyResolution / 2);
	}

	writeSpectrogramToImage(networkPredictions, "_Testing/Predictions/vocalPredictionTestTrackSpectrogram.png");
	

	// Get Samples and Write To Track
	vector<int16_t> testTrackOutputSamples = vocalSamples(trackName.data(), predictedTrackSpectrogram, config);
	writeToWAV("_Testing/Predictions/vocalPredictionTestTrack.wav", testTrackOutputSamples);
}
void testTrainOutputs(audioFileConfig config) {
	config.startFileIndex = 1;
	config.songCount = 1;

	// get outputs
	vector<vector<float>> fullOutputs = generateOutputs(config);

	// Get Input Spectrogram (from first file only)
	vector<vector<float>> testTrackSpectrogram = generateInputs(config); // First track only, for testing
	vector<vector<float>> predictedTrackSpectrogram;

	int chunkCount = testTrackSpectrogram.size();
	int indexJump = config.samplesPerChunk / config.samplesPerOverlap; // to support overlap output, skip "duplicate" output chunks

	if (config.skipOverlapChunks) {
		indexJump = 1;
	}

	// Get Network Predictions and Add to Output Track
	for (int i = 0; i < chunkCount; i += indexJump) {
		vector<float> currentChunkPrection = fullOutputs[i];

		// Use step function if binary mask has been used
		if (config.useOutputBinaryMask) {
			for (int j = 0; j < currentChunkPrection.size(); j++) {
				if (currentChunkPrection[j] > 0.5f) {
					currentChunkPrection[j] = 1.0f;
				}
				else {
					currentChunkPrection[j] = 0.0f;
				}
			}
		}

		if (config.useSingleOutputValue) {
			// Recreate full chunk
			vector<float> fullChunkOutput(config.frequencyResolution / 2, currentChunkPrection[0]);
			currentChunkPrection = fullChunkOutput;
		}

		predictedTrackSpectrogram.push_back(currentChunkPrection);
	}

	// Get Samples and Write To Track
	vector<int16_t> testTrackOutputSamples = vocalSamples("inputs/1.mp3", predictedTrackSpectrogram, config);
	writeToWAV("_Testing/datasetTest.wav", testTrackOutputSamples);
}

pair<vector<vector<float>>, vector<vector<float>>> getSingleSongDataset(audioFileConfig config, int chunksPerSong, int songIndex) {
	vector<vector<float>> resultantInputs;
	vector<vector<float>> resultantOutputs;

	config.startFileIndex = songIndex; // Go onto next file

	vector<vector<float>> songInputs = generateInputs(config);
	vector<vector<float>> songOutputs = generateOutputs(config);

	// Make sure randomly taken samples are regularly distributed across the song
	int miniBatchSize = songInputs.size() / chunksPerSong;

	for (int j = 0; j < chunksPerSong; j++) {
		int currentIndex = (j * miniBatchSize) + (rand() % miniBatchSize);

		if (songOutputs[currentIndex].size() == 0) {
			continue;
		}

		//cout << songInputs.size() << " " << songOutputs.size() << " " << currentIndex << endl;
		resultantInputs.push_back(songInputs[currentIndex]);
		resultantOutputs.push_back(songOutputs[currentIndex]);
	}

	return make_pair(resultantInputs, resultantOutputs);
}
pair<vector<vector<float>>, vector<vector<float>>> generateAllSongDataSet(audioFileConfig config, int chunksPerSong, int startSong, int endSong) {
	vector<vector<float>> resultantInputs;
	vector<vector<float>> resultantOutputs;

	config.songCount = 1;
	cout << "Loaded Song: ";

	for (int i = startSong; i < endSong; i++) { // 100 song count		
		pair<vector<vector<float>>, vector<vector<float>>> result = getSingleSongDataset(config, chunksPerSong, i + 1);

		resultantInputs.insert(resultantInputs.end(), result.first.begin(), result.first.end());
		resultantOutputs.insert(resultantOutputs.end(), result.second.begin(), result.second.end());

		cout << i + 1 << ", ";
	}

	cout << "Dataset Size: " << resultantInputs.size() << endl;
	pair<vector<vector<float>>, vector<vector<float>>> resultantDataset = make_pair(resultantInputs, resultantOutputs);

	return resultantDataset;
}

// Image
vector<vector<float>> addCharacterToImage(vector<vector<float>> data, int character, int xMidpoint, int yMidpoint) {
	vector<vector<float>> characterPixels; // 5 x 5

	// Character pixels of numbers 1 -> 9 (on a 5x5 grid)
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