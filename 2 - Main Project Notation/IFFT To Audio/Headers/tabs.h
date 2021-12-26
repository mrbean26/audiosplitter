#ifndef TABS_H
#define TABS_H

#include <vector>
#include "graphics.h"
using namespace std;

#define TAB_EDGE_DISTANCE 0.03f
#define TAB_LINE_GAP 0.02f

#define TAB_TEXT_DISTANCE 0.015f

extern vector<GLuint> tabVAOs;
extern vector<GLuint> tabVBOs;

void tabsBegin(vector<int> stringCounts);
void drawTabLines(int index, float yOffset);

void drawTab(vector<vector<int>> noteFrets);

#endif // !TABS_H
