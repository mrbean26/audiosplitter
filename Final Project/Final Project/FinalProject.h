#ifndef FINAL_PROJECT_H
#define FINAL_PROJECT_H

#include "Headers/splitter.h"
#include "Headers/graphics.h"

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

	int loadButton;
	int saveButton;
	
	int loadingBarOne;
	int loadingBarTwo;

	texture unmuteTexture;
	texture muteTexture;
	vector<int> muteButtons;

	// splitter 
	splitter mainSplitter;
	future<void> currentSplitterThread;

	const char* loadFileExplorer();
	void splitFile();
};

#endif // !FINAL_PROJECT_H

