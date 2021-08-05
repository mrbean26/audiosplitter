#include "Headers/audio.h"
#include "Headers/fftw3.h"

int main() {
	int frequencyResolution = 128;
	int samplesPerChunk = 2048;

	int valuesPerBand = samplesPerChunk / frequencyResolution;

	// Load Fully Correct NeuralNet Output
	pair<vector<vector<float>>, float> correctOutput = spectrogramOutput("output.mp3", samplesPerChunk, samplesPerChunk, frequencyResolution);
	vector<int16_t> outputSamples = vocalSamples("input.mp3", samplesPerChunk, samplesPerChunk, correctOutput.first, correctOutput.second);
	
	// Write to WAV
	writeToWAV("outputWave.wav", outputSamples);

	system("pause");
	return 0;
}