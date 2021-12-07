#ifndef AUDIO_H
#define AUDIO_H

#include <vector>
#include <iostream>
using namespace std;

#include "Headers/fftw3.h"
pair<vector<vector<float>>, float> spectrogramOutput(const char* mp3Filename, int samplesPerChunk, int samplesPerStride, int frequencyResolution);
vector<int16_t> vocalSamples(const char* fullFileNameMP3, int samplesPerChunk, int samplesPerStride, vector<vector<float>> networkOutput, float specmax);
void writeToWAV(const char* fileName, vector<int16_t> samples);

#endif // !AUDIO_H
