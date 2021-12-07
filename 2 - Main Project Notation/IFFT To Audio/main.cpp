#include "Headers/audio.h"
#include "Headers/fftw3.h"

int main() {
	int frequencyResolution = 64;
	int samplesPerChunk = 8192;
	
	float addedOutputError = 0.005f; // Make output not as perfect to make it more realistic
	float percentageFilter = 0.25f; // Filter bottom 25%

	// Load Fully Correct NeuralNet Output
	pair<vector<vector<float>>, float> correctOutput = spectrogramOutput("output.mp3", samplesPerChunk, samplesPerChunk, frequencyResolution);
	correctOutput = addSpectrogramError(correctOutput, addedOutputError);

	// Filter Output and Turn to Custom Note Format
	vector<vector<float>> filteredOutput = percentageFiltering(correctOutput.first, percentageFilter);
	vector<vector<int>> filteredNotes = returnNoteFormat(filteredOutput);

	system("pause");
	return 0;
}