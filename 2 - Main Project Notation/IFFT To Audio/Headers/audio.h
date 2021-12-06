#ifndef AUDIO_H
#define AUDIO_H

#include <vector>
#include <iostream>
using namespace std;

#include "Headers/fftw3.h"
pair<vector<vector<float>>, float> spectrogramOutput(const char* mp3Filename, int samplesPerChunk, int samplesPerStride, int frequencyResolution);
pair<vector<vector<float>>, float> addSpectrogramError(pair<vector<vector<float>>, float> original, float error);

#endif // !AUDIO_H
