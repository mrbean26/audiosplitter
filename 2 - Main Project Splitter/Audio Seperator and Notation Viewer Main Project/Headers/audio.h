#ifndef AUDIO_H
#define AUDIO_H

#include <vector>
#include <iostream>
using namespace std;

struct audioFileConfig {
	int samplesPerChunk = 2048;
	int samplesPerOverlap = 2048;

	int frequencyResolution = 128;
	int chunkBorder = 4;

	int startFileIndex = 1;
	int songCount = 1;

	float spectrogramEmphasis = 2.0f; // "No emphasis" = 1.0f

	bool useOutputBinaryMask = false;
	float binaryMaskThreshold = 0.025f;

	bool useNoisePrediction = false;

	bool useMelScale = true;
};

pair<vector<vector<float>>, float> spectrogramOutput(const char* mp3Filename, audioFileConfig audioConfig);
void writeSpectrogramToImage(vector<vector<float>> spectrogram, const char* fileName);

vector<int16_t> vocalSamples(const char* fullFileNameMP3, vector<vector<float>> networkOutput, audioFileConfig audioConfig);
void writeToWAV(const char* fileName, vector<int16_t> samples);

#endif // !AUDIO_H
