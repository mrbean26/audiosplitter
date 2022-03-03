#ifndef TABS_H
#define TABS_H

#include <vector>
#include "graphics.h"
#include "audio.h"
using namespace std;

#define TAB_EDGE_DISTANCE 0.03f
#define TAB_LINE_GAP 0.02f

#define TAB_TEXT_SIZE 1.25f // On an 1000px height screen
#define TAB_CHUNKS_PER_LINE 25 // On an 1000px width screen
#define TAB_SCROLL_RATE 0.025

#define PROGRESS_BAR_SIZE 0.089f // from image resolution ratio
#define LINE_LENGTH (1.0f - 2 * TAB_EDGE_DISTANCE)

class tabViewer {
public:
	audioObject* trackObjectPointer;
	tabViewer(vector<vector<int>> notes, vector<int> tunings, vector<int> maxFrets, int stringCount, int samplesPerChunk, int sampleRate, audioObject * trackAudio);

	float tabHeight = 0.0f;
	vector<GLuint> tabVAOs;
	vector<GLuint> tabVBOs;
	vector<GLuint> tabSizes;

	unsigned int tabShader;
	unsigned int imageShader;

	int samplesPerChunkProgress; // for progress bar calculations
	int sampleRateProgress;

	GLuint progressBarVAO;
	GLuint progressBarVBO;
	GLuint progressBarTexture;

	int tabStringCount = 0;
	vector<vector<int>> noteFrets;

	void tabsBegin(int stringCount);
	void drawTabLines(int index, float yOffset);
	
	float previousRuntime = 0.0f;
	float pausedTime = 0.0f;
	bool trackPaused = true;

	void pauseTrack();
	void resumeTrack();

	bool checkIfScroll();
	float currentOffset = 0.0f;

	int currentLineNumber = 0;
	mat4 getViewMatrix();

	void drawTab();

	void drawProgressBar();
};

#endif // !TABS_H
