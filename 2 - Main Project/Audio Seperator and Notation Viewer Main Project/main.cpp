#include "Headers/NeuralNetwork.h"

#include <iostream>
using namespace std;

#include "Headers/files.h"
#include "Headers/audio.h"

#include "Headers/matrices.h"

int main() {
	srand(time(NULL));
	audioFileConfig audioConfig = {
		2048, // samples per chunk
		2048, // samples per overlap

		64, // frequency res
		4, // chunk border

		1, // start file index
		1, // song count

		2.5f, // spectrogram emphasis, no emphasis = 1.0f

		false, // use binary mask for output
		0.025f, // binary mask threshold
	};

	// Train Network - One Song Training
	vector<vector<float>> inputSet = generateInputs(audioConfig);
	vector<vector<float>> outputSet = generateOutputs(audioConfig);

	int inputSize = inputSet[0].size();
	int outputSize = outputSet[0].size();

	// Big Train Algorithm
	int minLayers = 4;
	int maxLayers = 12;

	int minSize = 1;
	int maxSize = 4;

	int biasMin = 0;
	int biasMax = 3;

	NeuralNetwork::standardTrainConfig forStochastic = NeuralNetwork::standardTrainConfig();
	forStochastic.epochs = 1000;
	forStochastic.trainInputs = inputSet;
	forStochastic.trainOutputs = outputSet;
	forStochastic.momentum = 0.25f;
	forStochastic.learningRate = 1.0f;
	forStochastic.learningRateType = CYCLICAL_LEARNING_RATE;
	forStochastic.trainType = STOCHASTIC_GRADIENT_DESCENT;
	forStochastic.batchSize = 250;
	forStochastic.useStochasticDataset = true;
	forStochastic.stochasticDatasetSize = 250;

	for (int l = minLayers; l < maxLayers; l++) {
		for (int n = minSize; n < maxSize; n++) {
			for (int b = biasMin; b < biasMax; b++) {
				vector<int> layers = { inputSize };
				vector<int> biases = { b };
				vector<int> activations = { TANH };

				for (int i = 0; i < l; i++) {
					layers.push_back(n * 100);
					biases.push_back(b);
					activations.push_back(TANH);
				}

				layers.push_back(outputSize);
				biases.push_back(b);
				activations.push_back(TANH);

				// Gradient Descent
				NeuralNetwork newStochasticNetwork = NeuralNetwork(layers, biases, activations);
				vector<float> stochasticErrors = newStochasticNetwork.train(forStochastic);

				

				// Natural Selection
				NeuralNetwork selectionNetwork = NeuralNetwork::trainNaturalSelectionMethod(forStochastic, layers, biases, activations);

				// Find lowest error of best selection network here
				int datasetCount = inputSet.size();
				int outputCount = outputSet[0].size();

				float selectionError = 0.0f;
				for (int d = 0; d < datasetCount; d++) {
					vector<float> prediction = selectionNetwork.predict(inputSet[d]);

					for (int o = 0; o < outputCount; o++) {
						selectionError += abs(prediction[o] - outputSet[d][o]);
					}
				}
				outputImageConfig imageConfig = outputImageConfig{
					stochasticErrors,

					1000,
					512,

					newStochasticNetwork,
					audioConfig,
					forStochastic,

					true,
					1000.0f,
					0.0f,
				};

				imageConfig.errors = stochasticErrors;
				imageConfig.network = newStochasticNetwork;
				imageConfig.trainConfig = forStochastic;

				if (selectionError < stochasticErrors[999]) {
					imageConfig.trainConfig.trainType = LEVENBERG_MARQUARDT;
					imageConfig.errors[999] = selectionError;
				}
								
				writeToImage(imageConfig);
			}
		}
	}
	
		 
	// Some Version of Backpropagation Here 	
	vector<int> layers = { inputSize, 128, 128, 128, 128, 128, 128, outputSize };
	vector<int> biases = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	vector<int> activations = { SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID };

	NeuralNetwork newNetwork = NeuralNetwork(layers, biases, activations);
	
	NeuralNetwork::standardTrainConfig newConfig = NeuralNetwork::standardTrainConfig();
	newConfig.trainInputs = inputSet;
	newConfig.trainOutputs = outputSet;

	newConfig.epochs = 1000;
	newConfig.learningRate = 0.4f;
	newConfig.momentum = 0.05f;
	newConfig.learningRateType = FIXED_LEARNING_RATE;

	newConfig.trainType = STOCHASTIC_GRADIENT_DESCENT;
	newConfig.batchSize = 250;

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