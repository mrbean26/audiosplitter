#ifndef AUDIO_H
#define AUDIO_H

#include <vector>
#include <iostream>
using namespace std;

#include "Headers/fftw3.h"
pair<vector<vector<float>>, float> spectrogramOutput(const char* mp3Filename, int samplesPerChunk, int samplesPerStride, int frequencyResolution);
pair<vector<vector<float>>, float> addSpectrogramError(pair<vector<vector<float>>, float> original, float error);

vector<vector<float>> percentageFiltering(vector<vector<float>> inputSpectrogram, float percentageMultiplier);
vector<vector<int>> returnNoteFormat(vector<vector<float>> filteredSpectrogram);
vector<vector<int>> notesToFrets(vector<vector<int>> notes, vector<int> tunings, vector<int> maxFrets);

#endif // !AUDIO_H
