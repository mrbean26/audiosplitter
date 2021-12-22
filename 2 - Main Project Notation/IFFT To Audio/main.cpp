#include "Headers/audio.h"
#include "Headers/graphics.h"

int main() {
	int frequencyResolution = 8192;
	int samplesPerChunk = 8192;
	
	float addedOutputError = 0.005f; // Make output not as perfect to make it more realistic
	float percentageFilter = 0.75f; // Filter bottom 10%

	vector<int> tunings = { 7, 12, 17, 22, 26, 31 }; // Guitar Standard Tuning
	vector<int> maxFrets = { 21, 21, 21, 21, 21, 21 };

	// Load Fully Correct NeuralNet Output
	pair<vector<vector<float>>, float> correctOutput = spectrogramOutput("330hz.mp3", samplesPerChunk, samplesPerChunk, frequencyResolution);
	correctOutput = addSpectrogramError(correctOutput, addedOutputError);

	// Filter Output and Turn to Custom Note Format
	vector<vector<float>> filteredOutput = percentageFiltering(correctOutput.first, percentageFilter);
	vector<vector<int>> filteredNotes = returnNoteFormat(filteredOutput);
	vector<vector<int>> noteFrets = notesToFrets(filteredNotes, tunings, maxFrets);
	
	for (int i = 0; i < 6; i++) {
		cout << noteFrets[0][i] << endl;
	}
	// Graphics Rendering
	if (!startOpenGL(window, 1280, 720)) {
		return -1;
	}
	textsBegin();

	vec2 textWidthHeight = vec2();
	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		vec2 textWidthHeight = renderText("Welcome To The Tab Viewer", vec2(640.0f, 360.0f) - (textWidthHeight / 2.0f), 1.0f, 1.0f, vec3(1.0f, 1.0f, 1.0f), fontCharacters);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	return 0;
}