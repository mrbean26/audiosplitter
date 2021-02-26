#ifndef AUDIO_H
#define AUDIO_H

#include <vector>
#include <iostream>
using namespace std;

vector<vector<float>> spectrogramOutput(const char* mp3Filename, int samplesPerChunk, int frequencyResolution, int zeroRange, bool output);

#endif // !AUDIO_H
