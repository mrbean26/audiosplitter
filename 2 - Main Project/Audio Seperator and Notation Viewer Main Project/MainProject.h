#ifndef MAIN_PROJECT_H
#define MAIN_PROJECT_H

#include "Headers/audio.h"
#include "Headers/NeuralNetwork.h"

#include "Headers/notation.h"
#include "Headers/tabs.h"

// Splitting
#define HIGH_QUALITY 0
#define FAST_QUALITY 1

#define STEMS_VOCAL_ALL 0
#define STEMS_ALL 1

#define STEM_VOCAL 0
#define STEM_BASS 1
#define STEM_DRUMS 2

vector<vector<float>> getNetworkPredictions(NeuralNetwork* network, vector<vector<float>> inputs, int stem, int quality);
void displayStems(vector<vector<vector<float>>> networkOutputs, string fileName, int quality, int width, int height);

#endif // !MAIN_PROJECT_H

