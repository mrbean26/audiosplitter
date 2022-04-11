#include "MainProject.h"
#include "Headers/graphics.h"

// Audio
#define AUDIO_SAMPLES_PER_CHUNK 512
#define AUDIO_FREQUENCY_RESOLUTION 512
#define AUDIO_CHUNK_BORDER 12

// Notation
#define AUDIO_PERCENTAGE_FILTER_THRESHOLD 0.75f

// Splitter
string getWeightDirectory(int stem, int quality) {
	string qualityDirectory = "lowQuality/";
	if (quality == HIGH_QUALITY) {
		qualityDirectory = "highQuality/";
	}

	string stemFilename = "";
	if (stem == STEM_VOCAL) {
		stemFilename = "vocals.txt";
	}
	if (stem == STEM_BASS) {
		stemFilename = "bass.txt";
	}
	if (stem == STEM_DRUMS) {
		stemFilename = "drums.txt";
	}

	return "networkWeights/" + qualityDirectory + stemFilename;
}
vector<vector<float>> getNetworkPredictions(NeuralNetwork* network, vector<vector<float>> inputs, int stem, int quality) {
	network->loadWeightsFromFile(getWeightDirectory(stem, quality));

	vector<vector<float>> outputPredictions;
	int inputCount = inputs.size();

	for (int i = 0; i < inputCount; i++) {
		vector<float> currentPrediction = network->predict(inputs[i]);
		outputPredictions.push_back(currentPrediction);
	}

	return outputPredictions;
}
vector<vector<float>> getNetworkInputs(string fileName, audioFileConfig audioConfig) {
	vector<vector<float>> result;
	
	// Load File Spectrogram From File
	pair<vector<vector<float>>, float> spectrogram = spectrogramOutput(fileName.data(), audioConfig);
	vector<vector<float>> fullAudioInput = spectrogram.first;

	// Create Chunk Border to Give Audio 'Context'
	int newFrequencyResolution = fullAudioInput[0].size();

	for (int i = audioConfig.chunkBorder; i < fullAudioInput.size() - audioConfig.chunkBorder - 1; i++) {
		vector<float> currentInput;

		for (int c = i - audioConfig.chunkBorder; c < i + audioConfig.chunkBorder + 1; c++) {
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
	
	return result;
}

vector<vector<vector<float>>> getNetworkOutputs(string fileName, int stemCount, int quality) {
	vector<vector<vector<float>>> networkPredictionStems;
	NeuralNetwork splitterNetwork;
	audioFileConfig splitterAudioConfig;

	// Define Parameters
	if (quality == HIGH_QUALITY) {
		vector<int> nodes = {}; // Add later
		vector<int> bias = {}; // Add later
		vector<int> activations = {};

		splitterNetwork = NeuralNetwork(nodes, bias, activations);

		// Audio
		splitterAudioConfig = {
			// Add later
		};
	}
	if (quality == FAST_QUALITY) {
		vector<int> nodes = {}; // Add later
		vector<int> bias = {}; // Add later
		vector<int> activations = {};

		splitterNetwork = NeuralNetwork(nodes, bias, activations);

		// Audio
		splitterAudioConfig = {
			// Add later
		};
	}

	// Get Inputs
	vector<vector<float>> networkInputs = getNetworkInputs(fileName, splitterAudioConfig);

	// Get Stems
	if (stemCount == STEMS_VOCAL_ALL) {
		vector<vector<float>> vocalPredictions = getNetworkPredictions(&splitterNetwork, networkInputs, STEM_VOCAL, quality);
		vector<vector<float>> remainingTrackPredictions;

		// Find remaining track
		int chunkCount = vocalPredictions.size();
		int outputCount = vocalPredictions[0].size();

		for (int i = 0; i < chunkCount; i++) {
			vector<float> currentChunk = vocalPredictions[i];

			for (int j = 0; j < outputCount; j++) {
				currentChunk[j] = 1.0f - currentChunk[j];
			}

			remainingTrackPredictions.push_back(currentChunk);
		}

		return { remainingTrackPredictions, vocalPredictions };
	}
	if (stemCount == STEMS_ALL) {
		vector<vector<float>> vocalPredictions = getNetworkPredictions(&splitterNetwork, networkInputs, STEM_VOCAL, quality);
		vector<vector<float>> bassPredictions = getNetworkPredictions(&splitterNetwork, networkInputs, STEM_BASS, quality);
		vector<vector<float>> drumsPredictions = getNetworkPredictions(&splitterNetwork, networkInputs, STEM_DRUMS, quality);
		vector<vector<float>> otherPredictions;

		// Find "other" stem
		int chunkCount = vocalPredictions.size();
		int outputCount = vocalPredictions[0].size();

		for (int i = 0; i < chunkCount; i++) {
			vector<float> currentChunk(outputCount);

			for (int j = 0; j < outputCount; j++) {
				currentChunk[j] = 1.0f - (vocalPredictions[i][j] + bassPredictions[i][j] + drumsPredictions[i][j]);
			}

			otherPredictions.push_back(currentChunk);
		}

		return { vocalPredictions, bassPredictions, drumsPredictions, otherPredictions };
	}
}

// UI
vector<int> interfaceButtons;
vector<texture> interfaceTextures;

void createInterfaceButtons() {
	// Load textures
	interfaceTextures.push_back(loadTexture("Assets/Images/vocalIcon.png"));
	interfaceTextures.push_back(loadTexture("Assets/Images/instrumentalIcon.png"));
	interfaceTextures.push_back(loadTexture("Assets/Images/bassIcon.png"));
	interfaceTextures.push_back(loadTexture("Assets/Images/drumsIcon.png"));
	interfaceTextures.push_back(loadTexture("Assets/Images/otherIcon.png"));

	// Create current stem icon (image)
	int stemIcon = createButton(vec2(0.4f), vec3(0.8f, 0.8f, 0.0f), false);
	allButtons[stemIcon].texture = interfaceTextures[0];
	interfaceButtons.push_back(stemIcon);

	// Create shift button
	int shiftButton = createButton(vec2(0.25f), vec3(0.9f, 0.8f, 0.0f), true);
	allButtons[shiftButton].texture = loadTexture("Assets/Images/ShiftArrow.png");
	interfaceButtons.push_back(shiftButton);
}
void interfaceButtonMainloop() {

	if (allButtons[interfaceButtons[1]].clickUp) {
		int currentStemIndex = mainNotationViewer.currentStem;
		int stemCount = mainNotationViewer.noteLengths.size();

		if (stemCount == 2) { // Vocals & Instrumental
			currentStemIndex = currentStemIndex + 1;
			
			if (currentStemIndex == 2) {
				currentStemIndex = 0;
				allButtons[interfaceButtons[0]].texture = interfaceTextures[0];
			}
			else {
				allButtons[interfaceButtons[0]].texture = interfaceTextures[1];
			}
		}

		if (stemCount == 4) {
			currentStemIndex = currentStemIndex + 1;

			if (currentStemIndex == 4) {
				currentStemIndex = 0;
				allButtons[interfaceButtons[0]].texture = interfaceTextures[0];
			}
			else {
				allButtons[interfaceButtons[0]].texture = interfaceTextures[currentStemIndex + 1];
			}
		}

		mainNotationViewer.currentStem = currentStemIndex;
		mainTabViewer.currentStem = currentStemIndex;
	}
}

// Notation & Tab Viewer
notationViewer mainNotationViewer;
tabViewer mainTabViewer;
vector<audioObject> mainAudioObjects;

vector<vector<vector<float>>> outputStemsToSpectrogram(vector<vector<vector<float>>> networkOutputs, string fileName, audioFileConfig audioConfig) {
	// Find full track spectrogram and remove chunk border
	pair<vector<vector<float>>, float> fullTrackSpectrogram = spectrogramOutput(fileName.data(), audioConfig);
	fullTrackSpectrogram.first.erase(fullTrackSpectrogram.first.begin(), fullTrackSpectrogram.first.begin() + audioConfig.chunkBorder);
	fullTrackSpectrogram.first.resize(fullTrackSpectrogram.first.size() - audioConfig.chunkBorder);

	// Multiply spectrograms together
	vector<vector<vector<float>>> resultantSpectrograms;

	int stemCount = networkOutputs.size();
	for (int i = 0; i < stemCount; i++) {
		int chunkCount = networkOutputs[i].size();
		int frequencyCount = networkOutputs[i][0].size();

		for (int j = 0; j < chunkCount; j++) {
			for (int k = 0; k < frequencyCount; k++) {
				float currentValue = networkOutputs[i][j][k];

				if (audioConfig.useNoisePrediction) {
					currentValue = 1.0f - currentValue;
				}

				currentValue = powf(currentValue, audioConfig.spectrogramEmphasis);

				currentValue = currentValue * fullTrackSpectrogram.first[j][k];
				networkOutputs[i][j][k] = currentValue;
			}
		}

		resultantSpectrograms.push_back(networkOutputs[i]);
	}

	return resultantSpectrograms;
}
void openGLMainloop() {
	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		mainTabViewer.drawTab();
		mainNotationViewer.drawNotation();

		interfaceMainloop();
		interfaceButtonMainloop();
		
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
}

void displayStems(vector<vector<vector<float>>> networkOutputs, string fileName, int quality, int width, int height) {
	audioFileConfig splitterAudioConfig;
	/*
	// Define Parameters
	if (quality == HIGH_QUALITY) {
		splitterAudioConfig = {
			// Add later
		};
	}
	if (quality == FAST_QUALITY) {
		splitterAudioConfig = {
			// Add later
		};
	}

	vector<vector<vector<float>>> networkSpectrograms = outputStemsToSpectrogram(networkOutputs, fileName, splitterAudioConfig);
	int stemCount = networkSpectrograms.size();

	// Change to notes
	vector<vector<vector<int>>> stemNoteFormats;

	for (int i = 0; i < stemCount; i++) {
		networkSpectrograms[i] = percentageFiltering(networkSpectrograms[i], AUDIO_PERCENTAGE_FILTER_THRESHOLD);
		stemNoteFormats.push_back(returnNoteFormat(networkSpectrograms[i]));
	}
	*/











	

	int stemCount = 4;
	vector<vector<vector<int>>> stemNoteFormats = { {{1}}, {{1}}, {{1}} , {{1}} };











	// Start graphics
	startOpenAL();
	if (!startOpenGL(window, width, height)) {
		return;
	}
	textsBegin();
	
	// Instrument Config
	instrumentConfig newInstrumentConfig;
	newInstrumentConfig.tunings = { 7, 12, 17, 22, 26, 31 }; // Guitar Standard Tuning
	newInstrumentConfig.maxFrets = { 21, 21, 21, 21, 21, 21 };
	newInstrumentConfig.stringCount = 6;

	// Initialise Audio, Tab & Notation
	vector<audioObject*> pointers;
	for (int i = 0; i < stemCount; i++) {
		mainAudioObjects.push_back(audioObject(stemNoteFormats[i], AUDIO_SAMPLES_PER_CHUNK, 44100));
		pointers.push_back(&mainAudioObjects[i]);
	}

	mainNotationViewer = notationViewer(stemNoteFormats, AUDIO_SAMPLES_PER_CHUNK, 44100, pointers);
	mainTabViewer = tabViewer(stemNoteFormats, newInstrumentConfig.tunings, newInstrumentConfig.maxFrets, newInstrumentConfig.stringCount, AUDIO_SAMPLES_PER_CHUNK, 44100, pointers);

	createInterfaceButtons();

	interfaceBegin();
	openGLMainloop();
}