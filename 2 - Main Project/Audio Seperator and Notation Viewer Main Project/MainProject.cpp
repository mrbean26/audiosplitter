#include "MainProject.h"
#include "Headers/graphics.h"

#define AUDIO_SAMPLES_PER_CHUNK 512
#define AUDIO_FREQUENCY_RESOLUTION 512
#define AUDIO_CHUNK_BORDER 12
#define AUDIO_PERCENTAGE_FILTER_THRESHOLD 0.75f

#define NETWORK_CONFIG_LAYERS { 6144, 6144, 6144, 6144, 6144, 6144, 2048, 2048, 2048, 2048, 2048, 2048, 682, 682, 682, 682, 682, 256 }
#define NETWORK_CONFIG_BIAS { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
#define NETWORK_CONFIG_ACTIVATION { SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID, SIGMOID }

MainProject::MainProject(const char* inputFilename, vector<string> networkWeightDirectory, int width, int height) {
	start(inputFilename, networkWeightDirectory, width, height);
}
void MainProject::start(const char* inputFilename, vector<string> networkWeightDirectory, int width, int height) {
	// Prequisite Variables
	audioConfig = {
		AUDIO_SAMPLES_PER_CHUNK,
		AUDIO_SAMPLES_PER_CHUNK,

		AUDIO_FREQUENCY_RESOLUTION, // frequency res
		AUDIO_CHUNK_BORDER, // chunk border

		1, // start file index
		1, // song count

		2.5f, // spectrogram emphasis, no emphasis = 1.0f

		false, // use binary mask for output
		0.1f, // binary mask threshold

		false, // use noise prediction
	};

	instrumentConfig.tunings = { 7, 12, 17, 22, 26, 31 }; // Standard tuning
	instrumentConfig.maxFrets = { 21, 21, 21, 21, 21, 21 }; // Max Frets
	instrumentConfig.stringCount = 6; // String Count
	
	// Generate Network Inputs & Outputs
	vector<vector<float>> networkInputs = generateNetworkInputs(inputFilename);
	initialiseNetwork();

	vector<vector<float>> vocalOutputs = getNetworkOutputs(networkInputs, networkWeightDirectory[0].data());
	vocalOutputs = percentageFiltering(vocalOutputs, AUDIO_PERCENTAGE_FILTER_THRESHOLD);
	vocalNotes = returnNoteFormat(vocalOutputs);
	vocalSound = audioObject(vocalNotes, AUDIO_SAMPLES_PER_CHUNK, sampleRate);
	vocalNotation = notationViewer(vocalNotes, AUDIO_SAMPLES_PER_CHUNK, sampleRate, &vocalSound);
	vocalTab = tabViewer(vocalNotes, instrumentConfig.tunings, instrumentConfig.maxFrets, instrumentConfig.stringCount, AUDIO_SAMPLES_PER_CHUNK, sampleRate, &vocalSound);

	vector<vector<float>> bassOutputs = getNetworkOutputs(networkInputs, networkWeightDirectory[1].data());
	bassOutputs = percentageFiltering(bassOutputs, AUDIO_PERCENTAGE_FILTER_THRESHOLD);
	bassNotes = returnNoteFormat(bassOutputs);
	bassSound = audioObject(bassNotes, AUDIO_SAMPLES_PER_CHUNK, sampleRate);
	bassNotation = notationViewer(bassNotes, AUDIO_SAMPLES_PER_CHUNK, sampleRate, &bassSound);
	bassTab = tabViewer(bassNotes, instrumentConfig.tunings, instrumentConfig.maxFrets, instrumentConfig.stringCount, AUDIO_SAMPLES_PER_CHUNK, sampleRate, &bassSound);

	vector<vector<float>> drumsOutputs = getNetworkOutputs(networkInputs, networkWeightDirectory[2].data());
	drumsOutputs = percentageFiltering(drumsOutputs, AUDIO_PERCENTAGE_FILTER_THRESHOLD);
	drumsNotes = returnNoteFormat(drumsOutputs);
	drumsSound = audioObject(drumsNotes, AUDIO_SAMPLES_PER_CHUNK, sampleRate);
	drumsNotation = notationViewer(drumsNotes, AUDIO_SAMPLES_PER_CHUNK, sampleRate, &drumsSound);
	drumsTab = tabViewer(drumsNotes, instrumentConfig.tunings, instrumentConfig.maxFrets, instrumentConfig.stringCount, AUDIO_SAMPLES_PER_CHUNK, sampleRate, &drumsSound);

	// Graphics and Audio
	startOpenAL();
	if (!startOpenGL(window, width, height)) {
		return;
	}
	textsBegin();

}

vector<vector<float>> MainProject::generateNetworkInputs(const char* inputFilename) {
	pair<vector<vector<float>>, float> spectrogramChunks = spectrogramOutput(inputFilename, audioConfig);
	vector<vector<float>> result;

	int newFrequencyResolution = spectrogramChunks.first[0].size();
	for (int i = audioConfig.chunkBorder; i < spectrogramChunks.first.size() - audioConfig.chunkBorder; i++) {
		vector<float> currentInput;

		for (int c = i - audioConfig.chunkBorder; c < i + audioConfig.chunkBorder; c++) {
			for (int f = 0; f < newFrequencyResolution; f++) {
				float value = spectrogramChunks.first[i][f];

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
vector<vector<float>> MainProject::getNetworkOutputs(vector<vector<float>> inputs, string weightDirectory) {
	// Load Weights
	splitterNetwork.loadWeightsFromFile(weightDirectory);
	vector<vector<float>> result;

	int inputCount = inputs.size();
	for (int i = 0; i < inputCount; i++) {
		vector<float> output = splitterNetwork.predict(inputs[i]);
		result.push_back(output);
	}

	return result;
}

void MainProject::glfwMainloop() {
	int frame = 0;
	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if (frame == 0) {
			vocalNotation.pausedTime = glfwGetTime();
			vocalTab.pausedTime = glfwGetTime();

			bassNotation.pausedTime = glfwGetTime();
			bassTab.pausedTime = glfwGetTime();

			drumsNotation.pausedTime = glfwGetTime();
			drumsTab.pausedTime = glfwGetTime();
		}

		vocalTab.drawTab();

		glfwSwapBuffers(window);
		glfwPollEvents();
		frame++;
	}

	glfwDestroyWindow(window);
	glfwTerminate();
}

void MainProject::initialiseNetwork() {
	vector<int> layers = NETWORK_CONFIG_LAYERS;
	vector<int> bias = NETWORK_CONFIG_BIAS;
	vector<int> activations = NETWORK_CONFIG_ACTIVATION;

	splitterNetwork = NeuralNetwork(layers, bias, activations);
}