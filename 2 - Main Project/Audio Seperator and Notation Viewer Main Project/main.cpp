#include "Headers/NeuralNetwork.h"

#include <iostream>
using namespace std;

#include "Headers/files.h"
#include "Headers/audio.h"

#include "Headers/matrices.h"






#include "Headers/NaiveBayes.h"

int currentBayesIndex = 0;
bool a(DATASET_ENTRY entry) {
	int size = entry.second.size();
	int count = 0;

	for (int i = 0; i < size; i++) {
		if (entry.second[i] > 0.5f) {
			count += 1;
		}
	}

	if (float(count) > float(size) / 2.0f) {
		return true;
	}
	return false;
}

bool b(DATASET_ENTRY entry) {
	if (entry.second[currentBayesIndex] > 0.5) {
		return true;
	}
	return false;
}



int main() {
	audioFileConfig audioConfig = {
		2048, // samples per chunk
		2048, // samples per overlap

		64, // frequency res
		4, // chunk border

		1, // start file index
		4, // song count

		2.5f, // spectrogram emphasis, no emphasis = 1.0f

		false, // use binary mask for output
		0.025f, // binary mask threshold
	};

	// Train Network - One Song Training
	vector<vector<float>> inputSet = generateInputs(audioConfig);
	vector<vector<float>> outputSet = generateOutputs(audioConfig);




	// Test naive bayes
	vector<bayesProbability> probabilities;

	for (int i = 0; i < outputSet[0].size(); i++) {
		currentBayesIndex = i;

		bayesProbability current = getProbabilities(make_pair(inputSet, outputSet), a, b);
		probabilities.push_back(current);
	}
	// predict
	float totalError = 0.0f;
	float lowestB = 0.0f;

	for (int i = 0; i < outputSet.size(); i++) {
		vector<float> currentFreqIncludingContext = inputSet[i]; // --- this includes context

		int contextParameter = (audioConfig.frequencyResolution / 2) * audioConfig.chunkBorder;
		vector<float> currentFreq(currentFreqIncludingContext.begin() + contextParameter, currentFreqIncludingContext.begin() + contextParameter + (audioConfig.frequencyResolution / 2));


		float predictionVocal = 1.0f;
		for (int j = 0; j < currentFreq.size(); j++) {
			
			if (currentFreq[j] > 0.5f) {
				predictionVocal *= naiveBayes(probabilities[j]);
			}
			lowestB = min(lowestB, lowestB);
		}

		vector<float> predictedChunk;

		if (predictionVocal > 0.9f) {
			predictedChunk = currentFreq;
			

		}
		else {
			vector<float> newvector(currentFreq.size());
			predictedChunk = newvector;
		}

		vector<float> actualChunk = outputSet[i];

		for (int j = 0; j < currentFreq.size(); j++) {
			totalError += abs(predictedChunk[j] - actualChunk[j]);
		}
	}

	cout << totalError << endl;



	int inputSize = inputSet[0].size();
	int outputSize = outputSet[0].size();

	vector<int> layers = { inputSize, 448, 384, 320, 256, 192, 128, outputSize };
	vector<int> biases = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	vector<int> activations = { SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID};

	NeuralNetwork newNetwork = NeuralNetwork(layers, biases, activations);
	
	NeuralNetwork::standardTrainConfig newConfig = NeuralNetwork::standardTrainConfig();
	newConfig.trainInputs = inputSet;
	newConfig.trainOutputs = outputSet;

	newConfig.epochs = 25000;
	newConfig.learningRate = 0.75f;
	newConfig.momentum = 0.125f;
	newConfig.learningRateType = CYCLICAL_LEARNING_RATE;

	newConfig.entireBatchEpochIntervals = 1000;
	newConfig.batchSize = 400;

	newConfig.trainType = STOCHASTIC_GRADIENT_DESCENT;

	vector<float> errors = newNetwork.train(newConfig);
	createOutputTestTrack(newNetwork, audioConfig);

	outputImageConfig imageConfig = outputImageConfig{
		errors,

		1000,
		512,

		newNetwork,
		audioConfig,
		newConfig,

		true,
		1000.0f,
		0.0f,
	};

	writeToImage(imageConfig);
	

	system("pause");
	return 0;
}