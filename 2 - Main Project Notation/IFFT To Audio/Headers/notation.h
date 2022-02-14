#ifndef NOTATION_H
#define NOTATION_H

#include <vector>
#include "graphics.h"
#include "audio.h"
using namespace std;

#define NOTATION_EDGE_DISTANCE 0.05f
#define NOTATION_LINE_GAP 0.03f
#define NOTATION_MAX_LEDGER_LINES 3

#define NOTATION_CHUNKS_PER_LINE 20.0 // On a 1000px width screen

#define NOTATION_SHARP_DISTANCE 0.015f
#define NOTATION_SHARP_SIZE 1.75f // On a 1000px height screen

#define NOTATION_NOTE_LINE_WIDTH 0.002f

#define NOTATION_BPM_TEXT_SIZE 1.25f // On a 1000px height screen

#define NOTATION_SCROLL_RATE 0.025

class notationViewer {
public:
	notationViewer(vector<vector<int>> notes, int samplesPerChunk, int sampleRate, audioObject* trackAudio);

	vector<bool> keySignature;
	vector<vector<pair<int, int>>> noteLengths;

	vector<GLuint> notationVAOs;
	vector<GLuint> notationVBOs;
	vector<GLuint> notationSizes;

	vector<GLuint> noteTextures;
	GLuint trebleClefTexture;

	vec2 notationNoteSize;

	unsigned int notationShader;
	unsigned int imageShader;

	void startNotationShaders();
	void notationBegin();

	void startTrebleClef();
	void drawTrebleClef(float yOffset);

	void startStaveLines();
	void drawStaveLines(float yOffset);

	void startBarLine();
	void drawBarLine(float xOffset, float yOffset);

	void startNotes();
	void startNoteLine();
	void drawSingularNote(vec2 noteRootPosition, float staveCenter, int noteDuration, bool sharpSign);

	// Returned vector is length 5 - a true represents the note is a sharp - eg index 1 = C#
	static vector<bool> findKey(vector<vector<int>> notes);
	void drawKeySignature(vector<bool> keySignature, float yOffset);

	int samplesPerChunkProgress; // for progress bar calculations
	int sampleRateProgress;

	float previousRuntime = 0.0f;
	float pausedTime = 0.0f;
	bool trackPaused = false;

	bool checkIfScroll();
	float currentOffset = 0.0f;

	int currentLineNumber = 0;
	mat4 getViewMatrix();

	static bool compareNoteChunks(vector<int> chunkOne, vector<int> chunkTwo);
	static vector<vector<int>> removeNoteRepetitions(vector<vector<int>> originalChunks);
	static vector<vector<pair<int, int>>> findNoteLengths(vector<vector<int>> noteChunks);

	void drawNotes(vector<vector<pair<int, int>>> notes, vector<bool> keySignature);
	void drawNotation();
};

#endif // !NOTATION_H
