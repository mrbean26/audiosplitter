#ifndef NOTATION_H
#define NOTATION_H

#include <vector>
#include "graphics.h"
#include "audio.h"
using namespace std;

#define NOTATION_EDGE_DISTANCE 0.05f
#define NOTATION_LINE_GAP 0.03f
#define NOTATION_MAX_LEDGER_LINES 3

#define STAVE_HEIGHT 4.0f * NOTATION_LINE_GAP
#define NOTATION_HEIGHT ((5.0f * NOTATION_LINE_GAP + NOTATION_EDGE_DISTANCE) + (2 * NOTATION_MAX_LEDGER_LINES * NOTATION_LINE_GAP))
#define TREBLE_CLEF_WIDTH STAVE_HEIGHT * 0.3672f

#define NOTE_SIZE NOTATION_LINE_GAP
#define NOTE_SIZE_WIDTH 1.294f * NOTE_SIZE

#define NOTATION_CHUNKS_PER_LINE 20.0 // On a 1000px width screen

#define NOTATION_SHARP_DISTANCE 0.015f
#define NOTATION_SHARP_SIZE 1.75f // On a 1000px height screen

#define NOTATION_NOTE_LINE_WIDTH 0.002f

#define NOTATION_BPM_TEXT_SIZE 1.25f // On a 1000px height screen

#define PROGRESS_BAR_SIZE 0.089f * STAVE_HEIGHT
#define NOTATION_SCROLL_RATE 0.025

class notationViewer {
public:
	audioObject* trackObjectPointer;
	notationViewer(vector<vector<int>> notes, int samplesPerChunk, int sampleRate, audioObject* trackAudio);

	vector<bool> keySignature;
	vector<vector<pair<int, int>>> noteLengths;

	vector<GLuint> notationVAOs;
	vector<GLuint> notationVBOs;
	vector<GLuint> notationSizes;

	vector<GLuint> noteTextures;
	GLuint trebleClefTexture;

	GLuint progressBarVAO;
	GLuint progressBarVBO;
	GLuint progressBarTexture;

	vec2 notationNoteSize;

	unsigned int notationShader;
	unsigned int imageShader;

	void startNotationShaders();
	void notationBegin();

	void startTrebleClef();
	void drawTrebleClef(float yOffset);

	void startStaveLines();
	void drawStaveLines(float yOffset);

	GLuint ledgerLineVAO;
	GLuint ledgerLineVBO;
	GLuint ledgerLineSize;

	void drawLedgerLines(float noteY, float staveY, float noteX);
	void drawLedgerLine(float xOffset, float yOffset);

	void startBarLine();
	void drawBarLine(float xOffset, float yOffset);

	void startNotes();
	void startNoteLine();
	void drawSingularNote(vec2 noteRootPosition, float staveCenter, int noteDuration, bool sharpSign);

	// Returned vector is length 5 - a true represents the note is a sharp - eg index 1 = C#
	static vector<bool> findKey(vector<vector<int>> notes);
	float getKeySignatureWidth();
	void drawKeySignature(vector<bool> keySignature, float yOffset);

	int samplesPerChunkProgress; // for progress bar calculations
	int sampleRateProgress;

	float previousRuntime = 0.0f;
	float pausedTime = 0.0f;
	bool trackPaused = true;

	void pauseTrack();
	void resumeTrack();

	bool checkIfScroll();
	float currentOffset = 0.0f;

	int currentLineNumber = 0;
	mat4 getViewMatrix();

	static bool compareNoteChunks(vector<int> chunkOne, vector<int> chunkTwo);
	static vector<vector<int>> removeNoteRepetitions(vector<vector<int>> originalChunks);
	static vector<vector<pair<int, int>>> findNoteLengths(vector<vector<int>> noteChunks);
	static vector<vector<int>> removeOutOfRangeNotes(vector<vector<int>> inputNotes);

	float getLineLength();
	void drawNotes(vector<vector<pair<int, int>>> notes, vector<bool> keySignature);
	void drawNotation();

	void startProgressBar();
	void drawProgressBar();
};

#endif // !NOTATION_H
