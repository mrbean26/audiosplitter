#include "Headers/audio.h"
#include "Headers/graphics.h"

#include "Headers/tabs.h"
#include "Headers/notation.h"

int main() {
	int frequencyResolution = 8192;
	int samplesPerChunk = 8192;
	
	float addedOutputError = 0.005f; // Make output not as perfect to make it more realistic
	float percentageFilter = 0.75f; // Filter bottom 10%

	// Load Fully Correct NeuralNet Output
	pair<vector<vector<float>>, float> correctOutput = spectrogramOutput("ascendingStrings.mp3", samplesPerChunk, samplesPerChunk, frequencyResolution);
	correctOutput = addSpectrogramError(correctOutput, addedOutputError);

	// Filter Output and Turn to Custom Note Format
	vector<vector<float>> filteredOutput = percentageFiltering(correctOutput.first, percentageFilter);
	vector<vector<int>> filteredNotes = returnNoteFormat(filteredOutput);

	// Initialization
	instrumentConfig newInstrumentConfig;
	newInstrumentConfig.tunings = { 7, 12, 17, 22, 26, 31 }; // Guitar Standard Tuning
	newInstrumentConfig.maxFrets = { 21, 21, 21, 21, 21, 21 };
	newInstrumentConfig.stringCount = 6;

	saveNoteFormat({ make_pair(newInstrumentConfig, filteredNotes) }, "outputNotes.audio");
	vector<pair<instrumentConfig, vector<vector<int>>>> loadedNotes = loadNoteFormat("outputNotes.audio");

	// Graphics Rendering
	startOpenAL();
	if (!startOpenGL(window, 1280, 720)) {
		return -1;
	}

	audioObject newAudioObject = audioObject(filteredNotes, samplesPerChunk, 44100);
	notationViewer newNotationViewer = notationViewer(loadedNotes[0].second, samplesPerChunk, 44100, &newAudioObject);
	tabViewer newTabViewer = tabViewer(loadedNotes[0].second, loadedNotes[0].first.tunings, loadedNotes[0].first.maxFrets, loadedNotes[0].first.stringCount, samplesPerChunk, 44100, &newAudioObject);

	textsBegin();
	//newAudioObject.play();

	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//newTabViewer.drawTab();
		newNotationViewer.drawNotation();

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	return 0;
}