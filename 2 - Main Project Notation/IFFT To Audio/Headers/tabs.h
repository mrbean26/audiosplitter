#ifndef TABS_H
#define TABS_H

#include <vector>
#include "graphics.h"
using namespace std;

#define TAB_EDGE_DISTANCE 0.03f
#define TAB_LINE_GAP 0.02f

#define TAB_TEXT_SIZE 1.75f // On an 1000px height screen
#define TAB_CHUNKS_PER_LINE 40 // On an 1000px width screen

extern vector<GLuint> tabVAOs;
extern vector<GLuint> tabVBOs;

void tabsBegin(vector<int> stringCounts);
void drawTabLines(int index, float yOffset);

void drawTab(vector<vector<int>> noteFrets);

#endif // !TABS_H
