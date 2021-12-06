#include "Headers/NeuralNetwork.h"

#include <iostream>
using namespace std;

#include "Headers/files.h"
#include "Headers/audio.h"

int main() {
	srand(time(NULL));
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

	vector<vector<float>> inputSet = generateInputs(audioConfig);
	vector<vector<float>> outputSet = generateOutputs(audioConfig);
	cout << "Complete Dataset Size: " << inputSet.size() << endl;

	NeuralNetwork::standardTrainConfig newConfig = NeuralNetwork::standardTrainConfig();
	newConfig.epochs = 5;

	newConfig.population = 10;
	newConfig.parentCount = 2;

	newConfig.fitnessFunctionType = ABSOLUTE_ERROR;
	newConfig.parentSelectionMethod = TOP_PARENTS;

	newConfig.breedingMethod = WEIGHTED_PARENTS;
	newConfig.useChildMutation = true;

	newConfig.learningRate = 0.25f;
	newConfig.momentum = 0.0f;

	newConfig.useStochasticDataset = true;
	newConfig.stochasticDatasetSize = 1500;
	newConfig.useThreading = true;

	newConfig.selectionAllowedActivations = ACTIVATION_SIGMOID_ONLY;
	newConfig.selectionConvergenceCounter = 5;
	newConfig.selectionConvergenceValue = 10.0f;

	newConfig.selectionMinLayers = 3;
	newConfig.selectionMaxLayers = 18;
	
	newConfig.selectionMinNodes = 1;
	newConfig.selectionMaxNodes = 640;

	newConfig.selectionMinBias = 1;
	newConfig.selectionMaxBias = 5;

	newConfig.trainInputs = inputSet;
	newConfig.trainOutputs = outputSet;

	NeuralNetwork newNetwork = NeuralNetwork::architechtureNaturalSelection(newConfig);
	createOutputTestTrack(newNetwork, audioConfig);	

	system("pause");
	return 0;
}