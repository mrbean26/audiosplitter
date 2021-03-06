#include <iostream>
using namespace std;

#include "miniwav.h"
#include "fftw3.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

vector<vector<float>> spectrogramValuesFromSamples(vector<int> audioSamples, int samplesPerChunk, int samplesPerStride, int frequencyResolution) {
	vector<double> doubleAudioSamples(audioSamples.begin(), audioSamples.end());
	vector<vector<double>> spectrogramChunks;

	// Generate hanning window values, count = samplesPerChunk
	vector<double> hanningWindowValues;
	for (int i = 0; i < samplesPerChunk; i++) {
		double pi = 3.14159265358979323846;
		double currentCoefficient = double(i) / (double(samplesPerChunk) - 1);
		double cosValue = cos(2 * pi * currentCoefficient);

		double value = 0.5 * (1 - cosValue);
		hanningWindowValues.push_back(value);
	}

	// Split into chunks & apply hanning window function
	int sampleCount = doubleAudioSamples.size();
	for (int i = 0; i < sampleCount - samplesPerChunk; i += samplesPerStride) {
		vector<double> currentChunk(doubleAudioSamples.begin() + i, doubleAudioSamples.begin() + i + samplesPerChunk);

		for (int j = 0; j < samplesPerChunk; j++) {
			double currentValue = currentChunk[j] * hanningWindowValues[j];
			currentChunk[j] = currentValue;
		}

		spectrogramChunks.push_back(currentChunk);
	}

	// FFT each chunk 1D with FFTW
	fftw_complex* fftInputArray = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * samplesPerChunk);
	fftw_complex* fftOutput = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * samplesPerChunk);
	fftw_plan fftwPlan = fftw_plan_dft_1d(samplesPerChunk, fftInputArray, fftOutput, FFTW_FORWARD, FFTW_ESTIMATE);
	
	// Execute FFTW Plan and Convert Complex to Real
	int chunkCount = spectrogramChunks.size();
	for (int chunkNum = 0; chunkNum < chunkCount; chunkNum++) {
		for (int i = 0; i < samplesPerChunk; i++) {
			fftInputArray[i][0] = spectrogramChunks[chunkNum][i];
			fftInputArray[i][1] = 0;
		}

		fftw_execute(fftwPlan);

		for (int i = 0; i < samplesPerChunk; i++) {
			double real = fftOutput[i][0];
			double imaginary = fftOutput[i][1];

			spectrogramChunks[chunkNum][i] = sqrt(real * real + imaginary * imaginary);
		}
	}
	

	// Take Log Magnitude, Downsize Frequency Output with Average, and Take Max Value
	int valuesPerBand = samplesPerChunk / frequencyResolution;
	double maxValue = 0.0;

	for (int chunkNum = 0; chunkNum < chunkCount; chunkNum++) {
		vector<double> resultantArray;
		
		for (int i = 0; i < samplesPerChunk / 2; i += valuesPerBand) {
			double accumulativeValue = 0.0;

			for (int j = 0; j < valuesPerBand; j++) {
				double currentValue = abs(spectrogramChunks[chunkNum][i + j]);
				if (currentValue > 0) {
					currentValue = log(currentValue);
				}

				accumulativeValue = accumulativeValue + currentValue;
			}

			accumulativeValue = accumulativeValue / valuesPerBand;
			accumulativeValue = pow(1.5, accumulativeValue);
			maxValue = max(maxValue, accumulativeValue);
			
			resultantArray.push_back(accumulativeValue);
		}

		spectrogramChunks[chunkNum] = resultantArray;
	}
	// Change Values from Range 0 to 1
	vector<vector<float>> result;
	int newSamplesPerChunk = spectrogramChunks[0].size();

	for (int chunkNum = 0; chunkNum < chunkCount; chunkNum++) {
		for (int i = 0; i < newSamplesPerChunk; i++) {
			spectrogramChunks[chunkNum][i] = spectrogramChunks[chunkNum][i] / maxValue;
		}

		vector<float> currentVector(spectrogramChunks[chunkNum].begin(), spectrogramChunks[chunkNum].end());
		result.push_back(currentVector);
	}

	// Return as vector<vector<float>>
	return result;
}

int main() {
	wavData audioData = readWAVData("scartissue.wav");
	vector<vector<float>> spectrogramOutput = spectrogramValuesFromSamples(audioData.data, 16384, 16384 / 2, 128);

	int width = spectrogramOutput.size();
	int height = spectrogramOutput[0].size();
	int channelCount = 3;

	unsigned char * data = new unsigned char[width * height * channelCount];
	int index = 0;

	for (int y = height - 1; y >= 0; y--) {
		for (int x = 0; x < width; x++) {
			int intColour = int(255.0 * spectrogramOutput[x][y]);

			data[index++] = intColour;
			data[index++] = intColour;
			data[index++] = intColour;
		}
	}

	stbi_write_jpg("output.jpg", width, height, channelCount, data, width * sizeof(int));

	system("pause");
	return 0;
}