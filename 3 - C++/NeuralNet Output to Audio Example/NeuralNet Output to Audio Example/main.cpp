#include "Headers/audio.h"
#include "Headers/fftw3.h"

#include "Headers/audio.h"

int main() {
	// Prequisite Variables
	int samplesPerChunk = 8192;
	int samplesPerOverlap = samplesPerChunk; // no overlap - currently overlap is not implemented
	int frequencyResolution = 64;
	int chunkBorder = 20;
	
	float vocalThreshold = 0.1f;

	// 1 - Load Full Track into chunks of complex - no log magnitude, converting to real, etc.
	vector<fftw_complex*> fullTrackComplex = getFullTrackComplex("exampleInput.mp3", samplesPerChunk, samplesPerOverlap);

	// 2 - Load vocal output as 1 / 0 spectrogram with threshold
	vector<vector<float>> vocalSpectrogram = spectrogramOutput("exampleOutput.mp3", samplesPerChunk, samplesPerOverlap, frequencyResolution);

	for (int i = 0; i < vocalSpectrogram.size(); i++) {
		for (int j = 0; j < vocalSpectrogram[0].size(); j++) {
			float value = vocalSpectrogram[i][j];

			if (value >= vocalThreshold) {
				vocalSpectrogram[i][j] = 1.0f;
			}
			else {
				vocalSpectrogram[i][j] = 0.0f;
			}
		}
	}

	// 3 - Remove complex and real values where vocal output = 0
	// 4 - IFFT from FFTW with complex array to vector<double>
	// 5 - Run through backwards hanning window
	// 5 - Optional - write to MP3 or other audio file

	system("pause");
	return 0;
}