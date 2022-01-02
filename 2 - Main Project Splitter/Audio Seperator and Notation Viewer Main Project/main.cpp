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
	
	newConfig.epochs = 10;
	newConfig.learningRate = 0.25f;
	newConfig.momentum = 0.0f;

	newConfig.trainInputs = inputSet;
	newConfig.trainOutputs = outputSet;

	newConfig.naturalSelection.population = 10;
	newConfig.naturalSelection.parentCount = 2;

	newConfig.naturalSelection.fitnessFunctionType = ABSOLUTE_ERROR;
	newConfig.naturalSelection.parentSelectionMethod = TOP_PARENTS;

	newConfig.naturalSelection.breedingMethod = WEIGHTED_PARENTS;
	newConfig.naturalSelection.useChildMutation = true;

	newConfig.naturalSelection.useStochasticDataset = true;
	newConfig.naturalSelection.stochasticDatasetSize = 1500;
	newConfig.naturalSelection.useThreading = true;

	newConfig.naturalSelection.selectionAllowedActivations = ACTIVATION_SIGMOID_ONLY;
	newConfig.naturalSelection.selectionConvergenceCounter = 5;
	newConfig.naturalSelection.selectionConvergenceValue = 10.0f;

	newConfig.naturalSelection.selectionMinLayers = 3;
	newConfig.naturalSelection.selectionMaxLayers = 18;
	
	newConfig.naturalSelection.selectionMinNodes = 1;
	newConfig.naturalSelection.selectionMaxNodes = 640;

	newConfig.naturalSelection.selectionMinBias = 1;
	newConfig.naturalSelection.selectionMaxBias = 5;

	NeuralNetwork newNetwork = NeuralNetwork::architechtureNaturalSelection(newConfig);
	createOutputTestTrack(newNetwork, audioConfig);	

	system("pause");
	return 0;
}