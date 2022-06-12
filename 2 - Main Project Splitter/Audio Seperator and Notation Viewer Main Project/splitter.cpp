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
		0.75f, // percentage of chunk needed to be 1

		true, // use noise reduction
		10, // chunk size
		8, // required chunk count
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
		predictionNetwork.loadWeightsFromFile("_trained_weights/1st_proper_train/");
	}
	if (STEM == STEM_BASS) {

	}
	if (STEM == STEM_DRUMS) {

	}
	
	currentLoadedStemWeights = STEM;
}
vector<vector<float>> splitter::predictTrack(vector<vector<float>> inputs) {
	// prequisite variables
	int inputCount = inputs.size();
	int inputStep = audioConfig.samplesPerChunk / audioConfig.samplesPerOverlap;
	vector<vector<float>> resultantPredictions;

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
	}

	// remove noise algorithm if applicable
	if (audioConfig.useNoiseReduction) {
		removeChunkPredictionNoise(resultantPredictions, audioConfig);
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
