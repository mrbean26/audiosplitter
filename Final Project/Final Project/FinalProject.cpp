#include "FinalProject.h"
#include "Headers/files.h"

#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <windows.h>
#include <string.h>
#include <iostream>
#include <filesystem>
#include <experimental/filesystem>

using namespace std;
using namespace experimental;

FinalProject::FinalProject(int width, int height) {
	startOpenGL(window, width, height);
	startOpenAL();
	interfaceBegin();

	mainSplitter = splitter(STEM_VOCAL);

	createInterfaceButtons();
	openGLMainloop();
}

void FinalProject::createInterfaceButtons() {
	// Title
	int TitleImage = createButton(vec2(1.5f) * vec2(1.0, 0.407), vec3(0.0f, 0.7f, 0.0f), false);
	allButtons[TitleImage].texture = loadTexture("Assets/Images/title.png");

	// Load
	loadButton = createButton(vec2(0.25f), vec3(-0.3f, 0.7f, 0.0f), true);
	allButtons[loadButton].texture = loadTexture("Assets/Images/load.png");
	allButtons[loadButton].colour = vec3(0.0f);

	// Save
	saveButton = createButton(vec2(0.25f), vec3(0.3f, 0.7f, 0.0f), true);
	allButtons[saveButton].texture = loadTexture("Assets/Images/save.png");
	allButtons[saveButton].colour = vec3(0.4f);

	// Track Icons
	int vocalIcon = createButton(vec2(0.3f), vec3(-0.9f, 0.1f, 0.0f), false);
	allButtons[vocalIcon].texture = loadTexture("Assets/Images/vocalIcon.png");

	int otherIcon = createButton(vec2(0.3f), vec3(-0.9f, -0.2f, 0.0f), false);
	allButtons[otherIcon].texture = loadTexture("Assets/Images/instrumentalIcon.png");

	// mute and unmute buttons
	muteTexture = loadTexture("Assets/Images/mute.png");
	unmuteTexture = loadTexture("Assets/Images/unmute.png");

	for (int i = 0; i < 2; i++) {
		int newButton = createButton(vec2(0.3f), vec3(0.9f, 0.1f - i * 0.3f, 0.0f), true);
		allButtons[newButton].texture = muteTexture;
		
		muteButtons.push_back(newButton);
		muted.push_back(false);
	}

	// Loading Bar
	loadingBarOne = createButton(vec2(6.5f, 0.1f), vec3(-0.0f, 0.1f, 0.0f), false);
	loadingBarTwo = createButton(vec2(6.5f, 0.1f), vec3(-0.0f, -0.2f, 0.0f), false);
	
	scrollBarBackground = createButton(vec2(6.5f, 0.05f), vec3(0.0f, 0.3f, 0.0f), false);
	scrollBar = createButton(vec2(0.1f, 0.2f), vec3(-0.8f, 0.3f, 0.0f), true);

	// Pause and play
	playTexture = loadTexture("Assets/Images/play.png");
	pauseTexture = loadTexture("Assets/Images/pause.png");

	playButton = createButton(vec2(0.2f) * vec2(1.0f, 1.6f), vec3(0.0f, -0.45f, 0.0f), true);
	allButtons[playButton].texture = playTexture;
	allButtons[playButton].colour = vec3(0.4f);
}
void FinalProject::interfaceButtonMainloop() {
	// Load
	if (allButtons[loadButton].clickUp) {
		currentSplitterThread = future<void>();
		currentSplitterThread = async(&FinalProject::splitFile, this);
	}

	// save
	if (allButtons[saveButton].clickUp) {
		currentSplitterThread = future<void>();
		currentSplitterThread = async(&FinalProject::saveSamplesToFile, this);
	}

	// mute
	if (allButtons[muteButtons[0]].clickUp) {
		muted[0] = !muted[0];

		if (!muted[0]) {
			allButtons[muteButtons[0]].texture = muteTexture;
			alSourcef(sourceVocals, AL_GAIN, 1.0f);
		}
		else {
			allButtons[muteButtons[0]].texture = unmuteTexture;
			alSourcef(sourceVocals, AL_GAIN, 0.0f);
		}
	}
	if (allButtons[muteButtons[1]].clickUp) {
		muted[1] = !muted[1];

		if (!muted[1]) {
			allButtons[muteButtons[1]].texture = muteTexture;
			alSourcef(sourceInstrumentals, AL_GAIN, 1.0f);
		}
		else {
			allButtons[muteButtons[1]].texture = unmuteTexture;
			alSourcef(sourceInstrumentals, AL_GAIN, 0.0f);
		}
	}

	if (allButtons[playButton].clickUp) {
		paused = !paused;

		if (!paused) {
			alSourcePlay(sourceVocals);
			alSourcePlay(sourceInstrumentals);

			allButtons[playButton].texture = pauseTexture;
		}
		else {
			alSourcePause(sourceVocals);
			alSourcePause(sourceInstrumentals);
			
			allButtons[playButton].texture = playTexture;
		}
	}
}

const char* FinalProject::saveFileExplorer() {
	OPENFILENAME ofn;

	char szFileName[MAX_PATH] = { 0 };

	ZeroMemory(&ofn, sizeof(ofn));

	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = NULL;
	ofn.lpstrFilter = (LPCWSTR)L"Splitter Output (.splittersamples)\0*.*\0";
	ofn.lpstrFile = (LPWSTR)szFileName;
	ofn.nMaxFile = MAX_PATH;
	ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;

	string a = "";
	if (GetSaveFileName(&ofn) == TRUE) {
		filesystem::path myFile = ofn.lpstrFile;
		filesystem::path fullname = myFile.filename();

		for (int i = 0; i < MAX_PATH; i++) {
			if (szFileName[i] > 0) {
				a += szFileName[i];
			}

		}
	}

	a = a + ".splittersamples";
	return a.data();
}
const char* FinalProject::loadFileExplorer() {
	OPENFILENAME ofn = { 0 };
	TCHAR szFile[MAX_PATH] = { 0 };
	// Initialize remaining fields of OPENFILENAME structure
	ofn.lStructSize = sizeof(ofn);
	
	ofn.lpstrFile = szFile;
	ofn.nMaxFile = sizeof(szFile);
	
	ofn.nFilterIndex = 1;
	ofn.lpstrFileTitle = NULL;
	ofn.nMaxFileTitle = 0;
	ofn.lpstrInitialDir = NULL;
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
	ofn.lpstrFilter = (LPCWSTR)L"MP3 Files (*.mp3)\0\0Splitter Output (*.splittersamples)\0*.*\0";

	string a = "";
	if (GetOpenFileName(&ofn) == TRUE) {
		filesystem::path myFile = ofn.lpstrFile;
		filesystem::path fullname = myFile.filename();

		for (int i = 0; i < MAX_PATH; i++) {
			a += szFile[i];
		}
	}
	
	return a.data();
}

void FinalProject::removeTrack() {
	// change button colours
	allButtons[saveButton].colour = vec3(0.4f);
	allButtons[playButton].colour = vec3(0.4f);
	allButtons[loadButton].colour = vec3(0.4f);

	// clear samples
	mainSplitter.outputSamples.clear();
	generatedTracks = false;

	// reset loading bar
	mainSplitter.predictionsRequired = 0;
	mainSplitter.predictionsDone = 0;

	// delete audio objects
	alDeleteSources(1, &sourceVocals);
	alDeleteSources(1, &sourceInstrumentals);

	alDeleteBuffers(1, &bufferVocals);
	alDeleteBuffers(1, &bufferInstrumentals);

	allButtons[loadButton].colour = vec3(0.0f);
}
void FinalProject::splitFile() {
	allButtons[loadButton].colour = vec3(0.4f);

	string chosenFilename = loadFileExplorer();
	string fileExtension = chosenFilename.substr(chosenFilename.find(".") + 1);

	if (fileExtension == "mp3") {
		mainSplitter.splitStems(STEMS_VOCALS_BACKING, chosenFilename.data(), "");
	}
	if (fileExtension == "splittersamples") {
		loadSamplesFromFile(chosenFilename.data());
	}
	
	allButtons[loadButton].colour = vec3(0.0f);
	allButtons[playButton].colour = vec3(0.0f);
	allButtons[saveButton].colour = vec3(0.0f);
}

void FinalProject::openGLMainloop() {
	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		interfaceMainloop();
		interfaceButtonMainloop();
		
		updateLoadingBar();
		updateScrollBar();
		generateAudioObjects();

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
}
void FinalProject::updateLoadingBar() {
	float xScale = 0.0f;
	if (mainSplitter.predictionsRequired != 0) {
		xScale = (float(mainSplitter.predictionsDone) / float(mainSplitter.predictionsRequired)) * 6.5f;
	}

	float xPosition = -0.85f + (float(mainSplitter.predictionsDone) / float(mainSplitter.predictionsRequired)) * 0.85f;

	allButtons[loadingBarOne].scale = vec2(xScale, 0.1f);
	allButtons[loadingBarOne].position = vec3(xPosition, 0.1f, 0.0f);

	allButtons[loadingBarTwo].scale = vec2(xScale, 0.1f);
	allButtons[loadingBarTwo].position = vec3(xPosition, -0.2f, 0.0f);
}
void FinalProject::updateScrollBar() {
	if (generatedTracks) {
		// scrollbar button interactions
		if (allButtons[scrollBar].clickDown) {
			float minScrollbar = -0.8f;
			float maxScrollbar = 0.8f;

			// calculate mouse x as a proportion of screen
			float mouseXProportion = (mousePosX / (display_x / 2.0)) - 1.0f;

			// get new position
			float newXPosition = mouseXProportion;
			newXPosition = max(newXPosition, minScrollbar);
			newXPosition = min(newXPosition, maxScrollbar);

			allButtons[scrollBar].position.x = newXPosition;
		}
		if (allButtons[scrollBar].clickUp) {
			float currentScrollbarPosition = allButtons[scrollBar].position.x;
			float scrollbarProportion = (currentScrollbarPosition + 0.8f) / 1.6f;

			// set new track time
			float newSampleOffset = scrollbarProportion * float(mainSplitter.outputSamples[0].size());
			alSourcef(sourceVocals, AL_SAMPLE_OFFSET, newSampleOffset);
			alSourcef(sourceInstrumentals, AL_SAMPLE_OFFSET, newSampleOffset);
		}

		// update scrollbar naturally with track progression
		if (!allButtons[scrollBar].clickDown && !allButtons[scrollBar].clickUp) {
			float currentTrackSample;
			alGetSourcef(sourceVocals, AL_SAMPLE_OFFSET, &currentTrackSample);

			float currentTrackPercentage = currentTrackSample / float(mainSplitter.outputSamples[0].size());
			float newScrollbarXPosition = (currentTrackPercentage * 1.6f) - 0.8f;

			allButtons[scrollBar].position.x = newScrollbarXPosition;
		}
	}
}

void FinalProject::saveSamplesToFile() {
	allButtons[saveButton].colour = vec3(0.4f);
	allButtons[loadButton].colour = vec3(0.4f);

	const char* saveFileName = saveFileExplorer();
	ofstream outputFile(saveFileName, ios::out | ios::binary);

	// Sample Rate
	int16_t sampleRate = lastSeenFileSampleRate;
	outputFile.write((char*)&sampleRate, sizeof(sampleRate));
	outputFile << "DATA";

	// Write vocal samples
	int sampleCount = mainSplitter.outputSamples[0].size();
	
	for (int j = 0; j < 2; j++) {
		for (int i = 0; i < sampleCount; i++) {
			int16_t currentSample = mainSplitter.outputSamples[j][i];
			outputFile.write((char*)&currentSample, sizeof(currentSample));
		}

		outputFile << "CHUNK";
	}

	outputFile.close();

	allButtons[saveButton].colour = vec3(0.0f);
	allButtons[loadButton].colour = vec3(0.0f);
}
void FinalProject::loadSamplesFromFile(const char* fileName) {
	ifstream stream(fileName, std::ios::in | std::ios::binary);
	lastSeenFileSampleRate = stream.get();
	lastSeenFileSampleRate += (stream.get() << 8);

	// collect data
	vector<int> fileData;
	while (stream.good()) {
		fileData.push_back(stream.get());
	}
	
	bool finishedVocals = false;
	vector<int16_t> vocalSamples;
	vector<int16_t> backgroundSamples;

	// process samples
	int dataPoints = fileData.size();
	mainSplitter.predictionsRequired = dataPoints;

	for (int i = 4; i < dataPoints; i = i + 2) {
		// Check if first chunk has been read in
		if (char(fileData[i]) == 'C' && char(fileData[i + 1]) == 'H' && char(fileData[i + 2]) == 'U'){
			if (char(fileData[i + 3]) == 'N' && char(fileData[i + 4]) == 'K') {
				i = i + 5;
				finishedVocals = true;
			}
		}

		// Calculate current sample
		int intOne = fileData[i];
		int intTwo = fileData[i + 1];

		int16_t currentSample = intOne;
		currentSample += (intTwo << 8);

		if (!finishedVocals) {
			vocalSamples.push_back(currentSample);
		}
		if (finishedVocals) {
			backgroundSamples.push_back(currentSample);
		}

		// update loading bar
		mainSplitter.predictionsDone = i;
	}

	// add to splitter ready for audio processing
	mainSplitter.outputSamples.push_back(vocalSamples);
	mainSplitter.outputSamples.push_back(backgroundSamples);
}

void FinalProject::startOpenAL() {
	device = alcOpenDevice(NULL);
	context = alcCreateContext(device, NULL);
	alcMakeContextCurrent(context);
}
void FinalProject::generateAudioObjects() {
	if (generatedTracks || mainSplitter.outputSamples.size() != 2) {
		return;
	}

	// generate vocal audio
	alGenBuffers(1, &bufferVocals);
	alBufferData(bufferVocals, AL_FORMAT_MONO16, &mainSplitter.outputSamples[0][0], mainSplitter.outputSamples[0].size() * sizeof(ALshort), lastSeenFileSampleRate);
	
	alGenSources(1, &sourceVocals);
	alSourcei(sourceVocals, AL_BUFFER, bufferVocals);

	// generate instrumental
	alGenBuffers(1, &bufferInstrumentals);
	alBufferData(bufferInstrumentals, AL_FORMAT_MONO16, &mainSplitter.outputSamples[1][0], mainSplitter.outputSamples[1].size() * sizeof(ALshort), lastSeenFileSampleRate);

	alGenSources(1, &sourceInstrumentals);
	alSourcei(sourceInstrumentals, AL_BUFFER, bufferInstrumentals);

	generatedTracks = true;
	//mainSplitter.outputSamples.clear();
}