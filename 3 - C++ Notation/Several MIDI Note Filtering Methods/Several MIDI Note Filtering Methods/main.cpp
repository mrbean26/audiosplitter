#include <iostream>
#include <vector>
using namespace std;

#include <iomanip>
#include <fstream>

#include "Headers/fftw3.h"

#define MINIMP3_IMPLEMENTATION
#include "Headers/minimp3.h"
#include "Headers/minimp3_ex.h"

mp3dec_file_info_t loadAudioData(const char* mp3Filename) {
	mp3dec_t mp3Decoder;
	mp3dec_file_info_t fileInfo;

	if (mp3dec_load(&mp3Decoder, mp3Filename, &fileInfo, NULL, NULL)) {
		cout << "MP3 Load Error" << endl;
		return mp3dec_file_info_t();
	}

	return fileInfo;
}

vector<int> loadAudioSamples(mp3d_sample_t* buffer, int sampleCount, int channels) {
	if (channels == 1) {
		vector<int> result(sampleCount);

		for (int i = 0; i < sampleCount; i++) {
			result[i] = buffer[i];
		}
		return result;
	}
	if (channels == 2) {
		// Convert Stereo to Mono

		vector<int> result(sampleCount / 2);

		for (int i = 0; i < sampleCount / 2; i++) {
			result[i] = (buffer[i * 2] + buffer[i * 2 + 1]) / 2;
		}
		return result;
	}
}

vector<vector<float>> spectrogramOutput(const char* mp3Filename, int samplesPerChunk, int samplesPerStride, int frequencyResolution) {
	mp3dec_file_info_t audioData = loadAudioData(mp3Filename);
	vector<int> audioSamples = loadAudioSamples(audioData.buffer, audioData.samples, audioData.channels);
	delete[] audioData.buffer;
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

	// Downsize Frequency Output with Average, and Take Max Value
	int valuesPerBand = samplesPerChunk / frequencyResolution;
	double maxValue = 0.0;

	for (int chunkNum = 0; chunkNum < chunkCount; chunkNum++) {
		vector<double> resultantArray;

		for (int i = 0; i < samplesPerChunk; i += valuesPerBand) {
			double accumulativeValue = 0.0;

			for (int j = 0; j < valuesPerBand; j++) {
				double currentValue = spectrogramChunks[chunkNum][i + j];

				accumulativeValue = accumulativeValue + currentValue;
			}

			accumulativeValue = accumulativeValue / valuesPerBand;
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

		vector<float> currentVector(spectrogramChunks[chunkNum].begin(), spectrogramChunks[chunkNum].begin() + newSamplesPerChunk / 2);
		result.push_back(currentVector);
	}

	// Return as vector<vector<float>>
	return result;
}

vector<int16_t> vocalSamples(const char* fullFileNameMP3, int samplesPerChunk, int samplesPerStride, vector<vector<float>> networkOutput) {
	// Recreate full spectrogram
	int networkSubOutputSize = networkOutput[0].size();
	for (int i = 0; i < networkOutput.size(); i++) {
		for (int j = 0; j < networkOutput[i].size(); j++) { // Helpful for further isolating vocals
			double hannMultiplier = 0.5 * (1 - cos(2 * 3.141 * double(j) / double(networkSubOutputSize)));
			networkOutput[i][j] = networkOutput[i][j] * pow(hannMultiplier, 0.75); // index of 0.75 widens window at top, including more frequencies
		}

		vector<float> currentChunk = networkOutput[i];
		currentChunk.insert(currentChunk.end(), networkOutput[i].begin(), networkOutput[i].end());

		networkOutput[i] = currentChunk;
	}

	// IFFT Total
	mp3dec_file_info_t audioData = loadAudioData(fullFileNameMP3);
	vector<int> audioSamples = loadAudioSamples(audioData.buffer, audioData.samples, audioData.channels);

	int maxInitialSample = 0;
	for (int i = 0; i < audioSamples.size(); i++) {
		maxInitialSample = max(maxInitialSample, abs(audioSamples[i]));
	}

	vector<double> doubleAudioSamples(audioSamples.begin(), audioSamples.end());
	vector<vector<double>> spectrogramChunks;

	// Split into chunks
	int sampleCount = doubleAudioSamples.size();
	for (int i = 0; i < sampleCount - samplesPerChunk; i += samplesPerStride) {
		vector<double> currentChunk(doubleAudioSamples.begin() + i, doubleAudioSamples.begin() + i + samplesPerChunk);

		for (int j = 0; j < samplesPerChunk; j++) {
			double currentValue = currentChunk[j];
			currentChunk[j] = currentValue;
		}

		spectrogramChunks.push_back(currentChunk);
	}

	// FFT each chunk 1D with FFTW
	fftw_complex* fftInputArray = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * samplesPerChunk);
	fftw_complex* fftOutput = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * samplesPerChunk);
	fftw_plan fftwPlan = fftw_plan_dft_1d(samplesPerChunk, fftInputArray, fftOutput, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_plan fftwInversePlan = fftw_plan_dft_1d(samplesPerChunk, fftOutput, fftInputArray, FFTW_BACKWARD, FFTW_ESTIMATE);

	// Execute FFTW Plans
	int chunkCount = networkOutput.size();
	int frequencyResolution = networkOutput[0].size();
	int valuesPerBand = samplesPerChunk / frequencyResolution;

	vector<int16_t> resultantSamples;
	int maxNewSample = 0;

	for (int chunkNum = 0; chunkNum < chunkCount; chunkNum++) {
		for (int i = 0; i < samplesPerChunk; i++) {
			// Recreating full spectrogram here
			fftInputArray[i][0] = spectrogramChunks[chunkNum][i];
			fftInputArray[i][1] = 0;
		}

		fftw_execute(fftwPlan);

		for (int i = 0; i < samplesPerChunk; i += valuesPerBand) {
			for (int j = 0; j < valuesPerBand; j++) {
				fftOutput[i + j][0] = networkOutput[chunkNum][i / valuesPerBand] * fftOutput[i + j][0];
				fftOutput[i + j][1] = networkOutput[chunkNum][i / valuesPerBand] * fftOutput[i + j][1];
			}
		}

		fftw_execute(fftwInversePlan);

		for (int i = 0; i < samplesPerChunk; i++) {
			double normalisedReal = fftInputArray[i][0] / samplesPerChunk;
			double normalisedImag = fftInputArray[i][1] / samplesPerChunk;

			int16_t currentSample = normalisedReal;
			maxNewSample = max(maxNewSample, currentSample);
			resultantSamples.push_back(currentSample);
		}
	}

	fftw_destroy_plan(fftwPlan);
	fftw_destroy_plan(fftwInversePlan);
	fftw_free(fftInputArray);
	fftw_free(fftOutput);

	for (int i = 0; i < resultantSamples.size(); i++) {
		resultantSamples[i] = resultantSamples[i] * (double(maxInitialSample) / double(maxNewSample));
	}

	return resultantSamples;
}

void writeToWAV(const char* fileName, vector<int16_t> samples) {
	int sampleCount = samples.size();

	int16_t bitsPerSample = 16;
	const int32_t sampleRate = 44100;
	int16_t channelCount = 1;
	int32_t subChunk2Size = sampleCount * bitsPerSample * channelCount;
	int32_t chunkSize = subChunk2Size + 32;
	int16_t audioFormat = 1;
	int32_t subChunk1Size = 16;
	int32_t byteRate = sampleRate * channelCount * (bitsPerSample / 8);
	int16_t blockAlign = channelCount * (bitsPerSample / 8);

	// Open File
	std::ofstream outputFile(fileName, ios::out | ios::binary);

	// Write Header Info to File
	outputFile << "RIFF";
	outputFile.write((char*)&chunkSize, sizeof(chunkSize));
	outputFile << "WAVE";

	outputFile << "fmt ";
	outputFile.write((char*)&subChunk1Size, sizeof(subChunk1Size));
	outputFile.write((char*)&audioFormat, sizeof(audioFormat));
	outputFile.write((char*)&channelCount, sizeof(channelCount));
	outputFile.write((char*)&sampleRate, sizeof(sampleRate));
	outputFile.write((char*)&byteRate, sizeof(byteRate));
	outputFile.write((char*)&blockAlign, sizeof(blockAlign));
	outputFile.write((char*)&bitsPerSample, sizeof(bitsPerSample));

	// Data Chunk
	outputFile << "data";
	outputFile.write((char*)&subChunk2Size, sizeof(subChunk2Size));

	for (int i = 0; i < sampleCount; i++) {
		int16_t currentSample = samples[i];
		outputFile.write((char*)&currentSample, sizeof(currentSample));
	}
	// Close
	outputFile.close();
}

int main() {
	int samplesPerChunk = 1024;
	int frequencyResolution = 1024;

	// Generate Spectrogram
	vector<vector<float>> initialSpectrogram = spectrogramOutput("test2000hz.mp3", samplesPerChunk, samplesPerChunk, frequencyResolution);

	// Filter Notes Threshold
	float usedThreshold = 0.1f;
	float maxValueAcrossSpectrogram = 0.0f;

	for (int i = 0; i < initialSpectrogram.size(); i++) {
		for (int j = 0; j < initialSpectrogram[0].size(); j++) {
			maxValueAcrossSpectrogram = max(maxValueAcrossSpectrogram, initialSpectrogram[i][j]);
		}
	}

	usedThreshold = usedThreshold * maxValueAcrossSpectrogram;

	vector<vector<float>> finalSpectrogramThreshold;
	for (int i = 0; i < initialSpectrogram.size(); i++) {
		vector<float> resultantChunk;

		for (int j = 0; j < initialSpectrogram[0].size(); j++) {
			if (initialSpectrogram[i][j] < usedThreshold) {
				resultantChunk.push_back(0.0f);
			}
			else {
				resultantChunk.push_back(initialSpectrogram[i][j]);
			}
		}

		finalSpectrogramThreshold.push_back(resultantChunk);
	}
	
	// Filter Notes Exponent then Threshold
	vector<vector<float>> finalSpectrogramExponent;
	int exponent = 8;
	float multiple = 5.0f;

	usedThreshold = 0.1f;
	maxValueAcrossSpectrogram = 0.0f;

	for (int i = 0; i < initialSpectrogram.size(); i++) {
		for (int j = 0; j < initialSpectrogram[0].size(); j++) {
			maxValueAcrossSpectrogram = max(maxValueAcrossSpectrogram, initialSpectrogram[i][j]);
		}
	}
	
	usedThreshold = usedThreshold * maxValueAcrossSpectrogram;
	for (int i = 0; i < initialSpectrogram.size(); i++) {
		vector<float> resultantChunk;

		for (int j = 0; j < initialSpectrogram[0].size(); j++) {
			double value = pow(initialSpectrogram[i][j] * multiple, exponent);

			if (value < usedThreshold) {
				resultantChunk.push_back(0.0f);
			}
			else {
				resultantChunk.push_back(initialSpectrogram[i][j]);
			}
		}

		finalSpectrogramExponent.push_back(resultantChunk);
	}
	
	// Percentage Filtering (E.g take top 75% of each chunk)
	vector<vector<float>> finalSpectrogramPercentage;
	float percentageThreshold = 0.75f; // Take top 25% (1 - 0.75)

	for (int i = 0; i < initialSpectrogram.size(); i++) {
		vector<float> resultantChunk;

		float maxValue = 0.0f;
		for (int j = 0; j < initialSpectrogram[0].size(); j++) {
			maxValue = max(maxValue, initialSpectrogram[i][j]);
		}

		float threshold = maxValue * percentageThreshold;
		for (int j = 0; j < initialSpectrogram[0].size(); j++) {
			if (initialSpectrogram[i][j] > threshold) {
				resultantChunk.push_back(initialSpectrogram[i][j]);

				if (j > 2) { // Filter out some lower frequency 'noise' data
					// Works best with closest frequencyresolution and samples per chunk
					// Mutliply height of index in spectrogram (place in spectrogram) (first term)
					// By the maximum frequency (sample rate / 2) (second term)

					float predictedFrequency = (float(j) / float(initialSpectrogram[0].size())) * (44100.0f / 2.0f);
					cout << "Predicted Frequency: " << predictedFrequency << endl;

					// Note detection
					// Formula added to research

					// 55 hz base is at A
					int noteGap = log2f(predictedFrequency / 55.0f) * 12;
					noteGap = noteGap % 12;

					vector<string> notes = {"A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"};
					cout << "Predicted Note: " << notes[noteGap] << endl;
				}
			}
			else {
				resultantChunk.push_back(0.0f);
			}
		}

		finalSpectrogramPercentage.push_back(resultantChunk);
	}

	// Spectrogram to Samples
	vector<int16_t> outputSamplesThreshold = vocalSamples("vocals.mp3", samplesPerChunk, samplesPerChunk, finalSpectrogramThreshold);
	vector<int16_t> outputSamplesExponent = vocalSamples("vocals.mp3", samplesPerChunk, samplesPerChunk, finalSpectrogramExponent);
	vector<int16_t> outputSamplesPercentage = vocalSamples("vocals.mp3", samplesPerChunk, samplesPerChunk, finalSpectrogramPercentage);

	// Write to WAV
	writeToWAV("Threshold.wav", outputSamplesThreshold);
	writeToWAV("Exponent.wav", outputSamplesExponent);
	writeToWAV("Percentage.wav", outputSamplesPercentage);
	
	system("pause");
	return 0;
}