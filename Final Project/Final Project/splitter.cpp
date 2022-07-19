#include "Headers/splitter.h"
#include "Headers/files.h"

splitter::splitter() {}
splitter::splitter(int STEM) {
	// Get network
	getNetwork();
	getAudioConfig();

	loadStemWeights(STEM);
	outputCount = 64;
}

void splitter::getNetwork() {
	vector<int> nodes = { 1088, 750, 500, 300, 100, 64 };
	vector<int> bias(nodes.size(), 1);
	vector<int> activations(nodes.size(), SIGMOID);

	predictionNetwork = NeuralNetwork(nodes, bias, activations);
}
void splitter::getAudioConfig() {
	audioConfig = {
		1024, // samples per chunk
		1024, // samples per overlap

		128, // frequency res
		8, // chunk border

		1, // start file index
		3, // song count

		2.0f, // spectrogram emphasis, no emphasis = 1.0f

		true, // use binary mask for output
		0.1f, // binary mask threshold
		0.8f, // network output threshold binary

		false, // use noise prediction
		true, // use mel scale

		true, // skip chunk overlap

		false, // single output value
		0.5f, // percentage of chunk needed to be 1

		true, // use noise reduction
		5, // chunk size
		3, // required chunk count
		NOISE_REDUCTION_CHUNKS, // nosie type
	};
}

NeuralNetwork::standardTrainConfig splitter::getTrainConfig() {
	NeuralNetwork::standardTrainConfig newConfig = NeuralNetwork::standardTrainConfig();

	newConfig.trainType = GRADIENT_DESCENT;
	newConfig.epochs = 1000;

	newConfig.gradientDescent.learningRateType = FIXED_LEARNING_RATE;
	newConfig.learningRate = 0.01f;
	newConfig.momentum = 0.05f;

	// All song training
	pair<vector<vector<float>>, vector<vector<float>>> allSongDataset = generateAllSongDataSet(audioConfig, 1000, 0, 100);

	newConfig.trainInputs = allSongDataset.first;
	newConfig.trainOutputs = allSongDataset.second;

	return newConfig;
}
void splitter::trainNetwork(int STEM, const char* weightOutputDirectory) {
	NeuralNetwork::standardTrainConfig trainConfig = getTrainConfig();
	predictionNetwork.train(trainConfig);
	predictionNetwork.saveWeightsToFile(weightOutputDirectory);
}

void splitter::loadStemWeights(int STEM) {
	// Load weights from predefined directories
	if (STEM == STEM_VOCAL) {
		predictionNetwork.loadWeightsFromFile("_trained_weights/vocals/1st_proper_train_final/");
	}
	if (STEM == STEM_BASS) {
		predictionNetwork.loadWeightsFromFile("_trained_weights/bass/1st_proper_train/");
	}
	if (STEM == STEM_DRUMS) {
		predictionNetwork.loadWeightsFromFile("_trained_weights/drums/1st_proper_train/");
	}
	
	currentLoadedStemWeights = STEM;
}
vector<vector<float>> splitter::predictTrack(vector<vector<float>> inputs) {
	// prequisite variables
	int inputCount = inputs.size();
	int inputStep = audioConfig.samplesPerChunk / audioConfig.samplesPerOverlap;
	vector<vector<float>> resultantPredictions;

	predictionsDone = 0;
	predictionsRequired = inputCount / inputStep;

	// predict for each input
	for (int i = 0; i < inputCount; i += inputStep) {
		vector<float> currentPrediction = predictionNetwork.predict(inputs[i]);

		// binary mask if applicable
		if (audioConfig.useOutputBinaryMask) {
			for (int j = 0; j < outputCount; j++) {
				if (currentPrediction[j] > audioConfig.networkOutputThreshold) {
					currentPrediction[j] = 1.0f;
				}
				else {
					currentPrediction[j] = 0.0f;
				}
			}
		}

		resultantPredictions.push_back(currentPrediction);

		predictionsDone += 1;
	}

	// remove noise algorithm if applicable
	if (audioConfig.useNoiseReduction) {
		resultantPredictions = removeChunkPredictionNoise(resultantPredictions, audioConfig);
	}

	return resultantPredictions;
}
void splitter::predictTrackStemToFile(const char* inputFilename, int STEM, const char* outputFilename) {
	// load network weights if not loaded
	if (currentLoadedStemWeights != STEM) {
		loadStemWeights(STEM);
	}

	// predict using network
	vector<vector<float>> trackInput = generateSingleTrackInput(audioConfig, inputFilename);
	vector<vector<float>> predictedTrack = predictTrack(trackInput);

	// write to file
	vector<int16_t> testTrackOutputSamples = vocalSamples(inputFilename, predictedTrack, audioConfig);
	writeToWAV(outputFilename, testTrackOutputSamples);
}

vector<vector<float>> splitter::flipOutputVector(vector<vector<float>> input) {
	vector<vector<float>> result;

	int count = input.size();
	int subChunkSize = input[0].size();

	for (int i = 0; i < count; i++) {
		vector<float> currentChunk;

		for (int j = 0; j < subChunkSize; j++) {
			currentChunk.push_back(1.0f - input[i][j]);
		}

		result.push_back(currentChunk);
	}

	return result;
}
vector<vector<float>> splitter::addOutputVectors(vector<vector<float>> inputOne, vector<vector<float>> inputTwo) {
	vector<vector<float>> result;

	int count = inputOne.size();
	int subChunkSize = inputOne[0].size();

	for (int i = 0; i < count; i++) {
		vector<float> currentChunk;

		for (int j = 0; j < subChunkSize; j++) {
			if (inputOne[i][j] == 1.0f || inputTwo[i][j] == 1.0f) {
				currentChunk.push_back(1.0f);
			}
			else {
				currentChunk.push_back(0.0f);
			}
		}

		result.push_back(currentChunk);
	}

	return result;
}

void splitter::splitStems(int STEMS_CHOICE, const char* inputFilename, string outputDirectory) {
	if (STEMS_CHOICE == STEMS_VOCALS_BACKING) {
		if (currentLoadedStemWeights != STEM_VOCAL) {
			loadStemWeights(STEM_VOCAL);
		}

		cout << "Predicting Vocals..." << endl;
		vector<vector<float>> trackInput = generateSingleTrackInput(audioConfig, inputFilename);
		vector<vector<float>> predictedTrackVocal = predictTrack(trackInput);

		cout << "Predicting Other..." << endl;
		vector<vector<float>> predictedTrackOther = flipOutputVector(predictedTrackVocal);

		// write vocals
		vector<int16_t> testTrackOutputSamples = vocalSamples(inputFilename, predictedTrackVocal, audioConfig);
		writeToWAV((outputDirectory + "vocals.wav").data(), testTrackOutputSamples);

		// write backing
		testTrackOutputSamples = vocalSamples(inputFilename, predictedTrackOther, audioConfig);
		writeToWAV((outputDirectory + "other.wav").data(), testTrackOutputSamples);
	}
	if (STEMS_CHOICE == STEMS_ALL) {
		vector<vector<float>> trackInput = generateSingleTrackInput(audioConfig, inputFilename);

		// predict stems
		if (currentLoadedStemWeights != STEM_VOCAL) {
			loadStemWeights(STEM_VOCAL);
		}
		cout << "Predicting Vocals..." << endl;
		vector<vector<float>> predictedTrackVocal = predictTrack(trackInput);

		cout << "Predicting Bass..." << endl;
		loadStemWeights(STEM_BASS);
		vector<vector<float>> predictedTrackBass = predictTrack(trackInput);

		cout << "Predicting Drums..." << endl;
		loadStemWeights(STEM_DRUMS);
		vector<vector<float>> predictedTrackDrums = predictTrack(trackInput);

		// add stems
		cout << "Predicting Other..." << endl;
		vector<vector<float>> totalStems = addOutputVectors(predictedTrackVocal, predictedTrackBass);
		totalStems = addOutputVectors(totalStems, predictedTrackDrums);

		vector<vector<float>> otherPredictions = flipOutputVector(totalStems);

		// write stems
		vector<int16_t> testTrackOutputSamples = vocalSamples(inputFilename, predictedTrackVocal, audioConfig);
		writeToWAV((outputDirectory + "vocals.wav").data(), testTrackOutputSamples);

		testTrackOutputSamples = vocalSamples(inputFilename, predictedTrackBass, audioConfig);
		writeToWAV((outputDirectory + "bass.wav").data(), testTrackOutputSamples);

		testTrackOutputSamples = vocalSamples(inputFilename, predictedTrackDrums, audioConfig);
		writeToWAV((outputDirectory + "drums.wav").data(), testTrackOutputSamples);

		testTrackOutputSamples = vocalSamples(inputFilename, otherPredictions, audioConfig);
		writeToWAV((outputDirectory + "other.wav").data(), testTrackOutputSamples);
	}
}