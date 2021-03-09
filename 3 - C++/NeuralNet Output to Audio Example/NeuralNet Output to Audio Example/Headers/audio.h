#ifndef AUDIO_H
#define AUDIO_H

#include <vector>
#include <iostream>
using namespace std;

#include "fftw3.h"

vector<vector<float>> spectrogramOutput(const char* mp3Filename, int samplesPerChunk, int samplesPerStride, int frequencyResolution);
vector<fftw_complex*> getFullTrackComplex(const char* fileNameMP3, int samplesPerChunk, int samplesPerStride);

#endif // !AUDIO_H
