#ifndef AUDIO_H
#define AUDIO_H

#include <vector>
#include <iostream>
using namespace std;

#define NOISE_REDUCTION_CHUNKS 0
#define NOISE_REDUCTION_CONSECUTIVE 1

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
	float networkOutputThreshold = 0.8f;

	bool useNoisePrediction = false;
	bool useMelScale = true;

	bool skipOverlapChunks = true; // avoid generating unneccesary train data

	bool useSingleOutputValue = false;
	float singleOutputChunkThreshold = 0.25f; // percentage of chunk required to be 1

	bool useNoiseReduction = true;
	int noiseReductionChunkSize = 20;
	int noiseReductionRequiredChunks = 12;
	int noiseReductionType = NOISE_REDUCTION_CHUNKS;
};

extern int lastSeenFileSampleRate;
pair<vector<vector<float>>, float> spectrogramOutput(const char* mp3Filename, audioFileConfig audioConfig);
pair<vector<vector<float>>, float> spectrogramOutput(vector<vector<double>> samplesChunks, audioFileConfig audioConfig, int sampleRate);

void writeSpectrogramToImage(vector<vector<float>> spectrogram, const char* fileName);

vector<int16_t> vocalSamples(const char* fullFileNameMP3, vector<vector<float>> networkOutput, audioFileConfig audioConfig);
void writeToWAV(const char* fileName, vector<int16_t> samples);

#endif // !AUDIO_H
