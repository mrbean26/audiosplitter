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
	pair<vector<vector<float>>, float> correctOutput = spectrogramOutput("5sLow5sHigh.mp3", samplesPerChunk, samplesPerChunk, frequencyResolution);
	correctOutput = addSpectrogramError(correctOutput, addedOutputError);

	// Filter Output and Turn to Custom Note Format
	vector<vector<float>> filteredOutput = percentageFiltering(correctOutput.first, percentageFilter);
	vector<vector<int>> filteredNotes = returnNoteFormat(filteredOutput);
	vector<ALshort> noteSineWave = notesToWave(filteredNotes, samplesPerChunk, 44100);




	ALCdevice* device;
	ALCcontext* context;
	ALuint buffer, source;

	// Initialization
	device = alcOpenDevice(NULL);
	context = alcCreateContext(device, NULL);
	alcMakeContextCurrent(context);
	alGenBuffers(1, &buffer);

	alBufferData(buffer, AL_FORMAT_STEREO16, &noteSineWave[0], noteSineWave.size() * sizeof(ALshort), SAMPLING_HZ);
	alGenSources(1, &source);
	alSourcei(source, AL_BUFFER, buffer);
	alSourcei(source, AL_LOOPING, AL_FALSE);
	alSourcePlay(source);




	instrumentConfig newInstrumentConfig;
	newInstrumentConfig.tunings = { 7, 12, 17, 22, 26, 31 }; // Guitar Standard Tuning
	newInstrumentConfig.maxFrets = { 21, 21, 21, 21, 21, 21 };
	newInstrumentConfig.stringCount = 6;

	saveNoteFormat({ make_pair(newInstrumentConfig, filteredNotes) }, "outputNotes.audio");
	vector<pair<instrumentConfig, vector<vector<int>>>> loadedNotes = loadNoteFormat("outputNotes.audio");

	// Graphics Rendering
	if (!startOpenGL(window, 1280, 720)) {
		return -1;
	}
	notationViewer newNotationViewer = notationViewer(loadedNotes[0].second);
	tabViewer newTabViewer = tabViewer(loadedNotes[0].second, loadedNotes[0].first.tunings, loadedNotes[0].first.maxFrets, { loadedNotes[0].first.stringCount });

	textsBegin();
	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		newTabViewer.drawTab();
		//newNotationViewer.drawNotation();

		glfwSwapBuffers(window);
		glfwPollEvents();
	}




	alSourceStop(source);
	alDeleteSources(1, &source);
	alDeleteBuffers(1, &buffer);
	alcMakeContextCurrent(NULL);
	alcDestroyContext(context);
	alcCloseDevice(device);




	return 0;
}