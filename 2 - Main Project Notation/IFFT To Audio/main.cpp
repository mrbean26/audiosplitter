#include "Headers/audio.h"
#include "Headers/fftw3.h"

int main() {
	int frequencyResolution = 64;
	int samplesPerChunk = 8192;
	
	float addedOutputError = 0.005f; // Make output not as perfect to make it more realistic

	// Load Fully Correct NeuralNet Output
	pair<vector<vector<float>>, float> correctOutput = spectrogramOutput("output.mp3", samplesPerChunk, samplesPerChunk, frequencyResolution);
	correctOutput = addSpectrogramError(correctOutput, addedOutputError);

	system("pause");
	return 0;
}