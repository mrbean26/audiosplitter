#ifndef FINAL_PROJECT_H
#define FINAL_PROJECT_H

#include "Headers/splitter.h"
#include "Headers/graphics.h"

#include "AL/al.h"
#include "AL/alc.h"

#include <future>
#include <thread>

class FinalProject {
public:
	FinalProject(int width, int height);

	void createInterfaceButtons();
	void interfaceButtonMainloop();

	// Display
	void openGLMainloop();
	void updateLoadingBar();

	// Saving and Loading
	void saveSamplesToFile();
	void loadSamplesFromFile(const char* fileName);

	// Loading
	int loadButton;
	int saveButton;
	
	int loadingBarOne;
	int loadingBarTwo;

	// pause play
	int playButton;
	bool paused = true;

	texture playTexture;
	texture pauseTexture;

	// Muting
	texture unmuteTexture;
	texture muteTexture;

	vector<int> muteButtons;
	vector<bool> muted;

	// splitter 
	splitter mainSplitter;
	future<void> currentSplitterThread;

	const char* saveFileExplorer();
	const char* loadFileExplorer();
	void splitFile();

	// audio
	ALCdevice* device;
	ALCcontext* context;

	void startOpenAL();

	ALuint sourceVocals;
	ALuint bufferVocals;
	
	ALuint sourceInstrumentals;
	ALuint bufferInstrumentals;

	void generateAudioObjects();
	bool generatedTracks = false;
};

#endif // !FINAL_PROJECT_H

