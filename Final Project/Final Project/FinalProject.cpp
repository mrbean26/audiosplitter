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

	// Save
	saveButton = createButton(vec2(0.25f), vec3(0.3f, 0.7f, 0.0f), true);
	allButtons[saveButton].texture = loadTexture("Assets/Images/save.png");

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
	}

	// Loading Bar
	loadingBarOne = createButton(vec2(6.5f, 0.1f), vec3(-0.0f, 0.1f, 0.0f), false);
	loadingBarTwo = createButton(vec2(6.5f, 0.1f), vec3(-0.0f, -0.2f, 0.0f), false);
}
void FinalProject::interfaceButtonMainloop() {
	// Load
	if (allButtons[loadButton].clickUp) {
		currentSplitterThread = async(&FinalProject::splitFile, this);
	}
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
	ofn.lpstrFilter = (LPCWSTR)L"MP3 Files (*.mp3)\0*.*\0";;

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
void FinalProject::splitFile() {
	const char* chosenFilename = loadFileExplorer();
	mainSplitter.splitStems(STEMS_VOCALS_BACKING, chosenFilename, "");
}

void FinalProject::openGLMainloop() {
	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		interfaceMainloop();
		interfaceButtonMainloop();
		updateLoadingBar();

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
