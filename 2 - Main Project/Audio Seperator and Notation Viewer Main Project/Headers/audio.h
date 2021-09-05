#ifndef AUDIO_H
#define AUDIO_H

#include <vector>
#include <iostream>
using namespace std;

#include "files.h"

pair<vector<vector<float>>, float> spectrogramOutput(const char* mp3Filename, audioFileConfig audioConfig);
vector<int16_t> vocalSamples(const char* fullFileNameMP3, vector<vector<float>> networkOutput, audioFileConfig audioConfig);
void writeToWAV(const char* fileName, vector<int16_t> samples);

#endif // !AUDIO_H
