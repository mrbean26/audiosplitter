#ifndef NOTATION_H
#define NOTATION_H

#include <vector>
#include "graphics.h"
using namespace std;

#define NOTATION_EDGE_DISTANCE 0.05f
#define NOTATION_LINE_GAP 0.03f
#define NOTATION_MAX_LEDGER_LINES 3

#define NOTATION_CHUNKS_PER_LINE 20.0 // On a 1000px width screen

#define NOTATION_SHARP_DISTANCE 0.015f
#define NOTATION_SHARP_SIZE 1.75f // On a 1000px height screen

#define NOTATION_NOTE_LINE_WIDTH 0.002f

#define NOTATION_BPM_TEXT_SIZE 1.25f // On a 1000px height screen

extern vector<GLuint> notationVAOs;
extern vector<GLuint> notationVBOs;

void notationBegin();

// Returned vector is length 5 - a true represents the note is a sharp - eg index 1 = C#
vector<bool> findKey(vector<vector<int>> notes);
void drawKeySignature(vector<bool> keySignature, float yOffset);

bool compareNoteChunks(vector<int> chunkOne, vector<int> chunkTwo);
vector<vector<int>> removeNoteRepetitions(vector<vector<int>> originalChunks);

vector<vector<pair<int, int>>> findNoteLengths(vector<vector<int>> noteChunks);

void drawTrebleClef(float yOffset);
void drawBarLine(float xOffset, float yOffset);

void drawSingularNote(vec2 noteRootPosition, float staveCenter, int noteDuration, bool sharpSign);
void drawNotes(vector<vector<pair<int, int>>> notes, vector<bool> keySignature);

void drawStaveLines(float yOffset);
void drawNotation(vector<vector<pair<int, int>>> notes, vector<bool> keySignature);

#endif // !NOTATION_H
