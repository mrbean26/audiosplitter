#include "Headers/audio.h"
#include <iomanip>
#define MINIMP3_IMPLEMENTATION
#include "Headers/minimp3.h"
#include "Headers/minimp3_ex.h"

mp3dec_file_info_t loadSamples(const char* mp3Filename) {
	mp3dec_t mp3Decoder;
	mp3dec_file_info_t fileInfo;

	if (mp3dec_load(&mp3Decoder, mp3Filename, &fileInfo, NULL, NULL)) {
		cout << "MP3 Load Error" << endl;
		return mp3dec_file_info_t();
	}

	return fileInfo;
}

vector<vector<float>> spectrogramOutput(const char* mp3Filename, int samplesPerChunk, int frequencyResolution, int zeroRange){
	mp3dec_file_info_t audioData = loadSamples(mp3Filename);
	float maxFrequency = (float) audioData.hz / 2.0f;

	float waveStartTime = 0.0f;
	int maxValue = 0;
	int zeroCount = 0;

	int maxIndexChunkNum = (int)(audioData.samples / samplesPerChunk);
	vector<vector<float>> result = {};

	for (int chunkNum = 0; chunkNum < maxIndexChunkNum; chunkNum++) {
		int startIndex = chunkNum * samplesPerChunk;
		vector<float> currentChunkResult(frequencyResolution);

		int currentWaveMaxVolume = 0;
		zeroCount = 0;
		waveStartTime = 0;

		for (int sampleNum = startIndex + 1; sampleNum < startIndex + samplesPerChunk; sampleNum++) {
			int lastSample = audioData.buffer[sampleNum - 1];
			int currentSample = audioData.buffer[sampleNum];

			if (zeroCount > 0) {
				currentWaveMaxVolume = max(currentWaveMaxVolume, currentSample);
			}

			if (abs(currentSample) - zeroRange < 0) {
				continue;
			}

			if ((lastSample < 0 && currentSample > 0) || (lastSample > 0 && currentSample < 0)) {
				zeroCount = zeroCount + 1;

				if (zeroCount == 1) {
					waveStartTime = ((float)sampleNum / (float)audioData.hz);
				}

				if (zeroCount == 3) {
					float waveEndTime = ((float)sampleNum / (float)audioData.hz);
					float waveFrequency = 1.0f / (waveEndTime - waveStartTime);
					
					int usedIndex = (int)((waveFrequency / maxFrequency) * (frequencyResolution - 1));
					maxValue = max(maxValue, currentWaveMaxVolume);
					currentChunkResult[usedIndex] = currentWaveMaxVolume;

					zeroCount = 0;
					currentWaveMaxVolume = 0.0f;
				}
			}
		}

		result.push_back(currentChunkResult);
	}

	// make values 0 --> 1 for network
	for (int chunkNum = 0; chunkNum < maxIndexChunkNum; chunkNum++) {
		for (int frequencySample = 0; frequencySample < frequencyResolution; frequencySample++) {
			result[chunkNum][frequencySample] = result[chunkNum][frequencySample] / ((float)maxValue);
		}
	}

	return result;
}
