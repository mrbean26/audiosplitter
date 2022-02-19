#ifndef MAIN_PROJECT_H
#define MAIN_PROJECT_H

#include "Headers/audio.h"
#include "Headers/NeuralNetwork.h"

#include "Headers/notation.h"
#include "Headers/tabs.h"

class MainProject {
public:
	audioFileConfig audioConfig;
	instrumentConfig instrumentConfig;

	MainProject(const char* inputFilename, vector<string> networkWeightDirectory, int width, int height);
	void start(const char* inputFilename, vector<string> networkWeightDirectory, int width, int height);

	vector<vector<float>> generateNetworkInputs(const char* inputFilename);
	vector<vector<float>> getNetworkOutputs(vector<vector<float>> inputs, string weightDirectory);

	vector<vector<int>> vocalNotes;
	vector<vector<int>> bassNotes;
	vector<vector<int>> drumsNotes;

	audioObject vocalSound;
	audioObject bassSound;
	audioObject drumsSound;

	// Graphics
	notationViewer vocalNotation;
	notationViewer bassNotation;
	notationViewer drumsNotation;

	tabViewer vocalTab;
	tabViewer bassTab;
	tabViewer drumsTab;

	void glfwMainloop();

	NeuralNetwork splitterNetwork;
	void initialiseNetwork();
};

#endif // !MAIN_PROJECT_H

