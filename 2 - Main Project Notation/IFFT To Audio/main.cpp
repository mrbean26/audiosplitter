#include "Headers/audio.h"
#include "Headers/graphics.h"

#include "Headers/tabs.h"
#include "Headers/notation.h"

int main() {
	int frequencyResolution = 8192;
	int samplesPerChunk = 8192;
	
	float addedOutputError = 0.005f; // Make output not as perfect to make it more realistic
	float percentageFilter = 0.75f; // Filter bottom 10%

	vector<int> tunings = { 7, 12, 17, 22, 26, 31 }; // Guitar Standard Tuning
	vector<int> maxFrets = { 21, 21, 21, 21, 21, 21 };

	// Load Fully Correct NeuralNet Output
	pair<vector<vector<float>>, float> correctOutput = spectrogramOutput("147hzD.mp3", samplesPerChunk, samplesPerChunk, frequencyResolution);
	correctOutput = addSpectrogramError(correctOutput, addedOutputError);

	// Filter Output and Turn to Custom Note Format
	vector<vector<float>> filteredOutput = percentageFiltering(correctOutput.first, percentageFilter);
	vector<vector<int>> filteredNotes = returnNoteFormat(filteredOutput);

	saveNoteFormat(filteredNotes, 6, "outputNotes.audio");
	vector<vector<int>> loadedNotes = loadNoteFormat("outputNotes.audio");
	vector<bool> keySignature = findKey(loadedNotes);
	
	vector<vector<int>> noteFrets = notesToFrets(loadedNotes, tunings, maxFrets);

	// Graphics Rendering
	if (!startOpenGL(window, 1280, 720)) {
		return -1;
	}

	notationBegin();
	textsBegin();
	tabsBegin({ 6 });

	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//drawTab(noteFrets);
		drawNotation(loadedNotes, keySignature);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	return 0;
}