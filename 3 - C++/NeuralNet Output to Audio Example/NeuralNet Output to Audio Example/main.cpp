#include "Headers/audio.h"
#include "Headers/fftw3.h"

#include "Headers/audio.h"
#include <fstream>

int main() {
	// Prequisite Variables
	int samplesPerChunk = 8192;
	int samplesPerOverlap = samplesPerChunk; // no overlap - currently overlap is not implemented
	int frequencyResolution = 64;
	int chunkBorder = 20;
	
	float vocalThreshold = 0.1f;

	// 1 - Load Full Track into chunks of complex - no log magnitude, converting to real, etc.
	vector<fftw_complex*> fullTrackComplex = getFullTrackComplex("exampleInput.mp3", samplesPerChunk, samplesPerOverlap);
	// Where size of each fftw_complex* is samplesPerChunk

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

	// 3 - Recompile full spectrogram without 3/4 trimming
	vector<vector<float>> fullVocalOut;
	vector<vector<float>> halfVocalOut;

	int count = vocalSpectrogram.size();
	for (int i = 0; i < count; i++) {
		vector<float> currentChunk = vocalSpectrogram[i];
		reverse(vocalSpectrogram[i].begin(), vocalSpectrogram[i].end());

		currentChunk.insert(currentChunk.end(), vocalSpectrogram[i].begin(), vocalSpectrogram[i].end());
		vector<float> temporary = currentChunk;

		currentChunk.insert(currentChunk.end(), temporary.begin(), temporary.end());
		fullVocalOut.push_back(currentChunk);
	}

	// 4 - Remove complex values of track when corresponding fullVocalOut = 0
	int chunkCount = fullVocalOut.size();
	int valuesPerBand = fullVocalOut[0].size() / frequencyResolution;

	for (int chunkNum = 0; chunkNum < chunkCount; chunkNum++) {
		for (int band = 0; band < frequencyResolution; band++) {
			if (fullVocalOut[chunkNum][band] == 0.0f) {
				for (int subBandNum = 0; subBandNum < valuesPerBand; subBandNum++) {
					int index = band * valuesPerBand + subBandNum;
					
					fullTrackComplex[chunkNum][index][0] = 0.0;
					fullTrackComplex[chunkNum][index][1] = 0.0;
				}
			}
		}
	}

	// 5 - Generate Hanning
	vector<float> hanningWindow;
	for (int i = 0; i < samplesPerChunk; i++) {
		float pi = 3.14159265358979323846;
		float currentCoefficient = float(i) / (float(samplesPerChunk) - 1);
		float cosValue = cos(2 * pi * currentCoefficient);

		float value = 0.5 * (1 - cosValue);
		hanningWindow.push_back(value);
	}

	// 6 - Inverse Fourier Transform Each Chunk, Convert To Real and apply Hanning
	fftw_complex* fftInputArray = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * samplesPerChunk);
	fftw_complex* fftOutput = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * samplesPerChunk);
	fftw_plan fftwPlan = fftw_plan_dft_1d(samplesPerChunk, fftInputArray, fftOutput, FFTW_BACKWARD, FFTW_ESTIMATE);

	vector<vector<float>> realIFFT;
	for (int chunkNum = 0; chunkNum < chunkCount; chunkNum++) {
		fftInputArray = fullTrackComplex[chunkNum];
		fftw_execute(fftwPlan);
		
		vector<float> currentChunk;
		for (int i = 0; i < samplesPerChunk; i++) {
			double real = fftOutput[i][0];
			double imaginary = fftOutput[i][1];

			float value = sqrt(real * real + imaginary * imaginary);
			value = value / hanningWindow[i];

			currentChunk.push_back(value);
		}

		realIFFT.push_back(currentChunk);
	}

	// 7 - Change to vector<int> and combine
	vector<int> result;

	for (int chunkNum = 0; chunkNum < chunkCount; chunkNum++) {
		vector<int> current(realIFFT[chunkNum].begin(), realIFFT[chunkNum].end());
		result.insert(result.end(), current.begin(), current.end());
	}

	// 8 - Write to file
	for (int i = 0; i < 200; i++) {
		cout << result[i] << endl;
	}

	system("pause");
	return 0;
}