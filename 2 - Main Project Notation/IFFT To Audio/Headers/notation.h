#ifndef NOTATION_H
#define NOTATION_H

#include <vector>
#include "graphics.h"
using namespace std;

#define NOTATION_EDGE_DISTANCE 0.05f
#define NOTATION_LINE_GAP 0.03f
#define NOTATION_MAX_LEDGER_LINES 3

#define NOTATION_CLEF_SIZE 0.05f // On a 1000px width screen
#define NOTATION_CHUNKS_PER_LINE 40.0 // On a 1000px width screen

extern vector<GLuint> notationVAOs;
extern vector<GLuint> notationVBOs;

void notationBegin();

void drawStaveLines(float yOffset);
void drawNotation(vector<vector<int>> notes);

#endif // !NOTATION_H
