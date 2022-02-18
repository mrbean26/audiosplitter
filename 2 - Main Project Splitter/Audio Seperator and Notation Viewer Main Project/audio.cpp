#include "Headers/audio.h"
#include <iomanip>
#include <fstream>

#define MINIMP3_IMPLEMENTATION
#include "Headers/minimp3.h"
#include "Headers/minimp3_ex.h"

#include "Headers/files.h"
#include "Headers/fftw3.h"

// General
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

// Network
pair<vector<vector<float>>, float> spectrogramOutput(const char* mp3Filename, audioFileConfig audioConfig) {
	mp3dec_file_info_t audioData = loadAudioData(mp3Filename);
	vector<int> audioSamples = loadAudioSamples(audioData.buffer, audioData.samples, audioData.channels);
	delete[] audioData.buffer; 

	vector<double> doubleAudioSamples(audioSamples.begin(), audioSamples.end());
	vector<vector<double>> spectrogramChunks;

	// Generate hanning window values, count = samplesPerChunk
	vector<double> hanningWindowValues;
	for (int i = 0; i < audioConfig.samplesPerChunk; i++) {
		double pi = 3.14159265358979323846;
		double currentCoefficient = double(i) / (double(audioConfig.samplesPerChunk) - 1);
		double cosValue = cos(2 * pi * currentCoefficient);

		double value = 0.5 * (1 - cosValue);
		hanningWindowValues.push_back(value);
	}

	// Split into chunks & apply hanning window function
	int sampleCount = doubleAudioSamples.size();
	for (int i = 0; i < sampleCount - audioConfig.samplesPerChunk; i += audioConfig.samplesPerOverlap) {
		vector<double> currentChunk(doubleAudioSamples.begin() + i, doubleAudioSamples.begin() + i + audioConfig.samplesPerChunk);

		for (int j = 0; j < audioConfig.samplesPerChunk; j++) {
			double currentValue = currentChunk[j] * hanningWindowValues[j];
			currentChunk[j] = currentValue;
		}

		spectrogramChunks.push_back(currentChunk);
	}

	// FFT each chunk 1D with FFTW
	fftw_complex* fftInputArray = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * audioConfig.samplesPerChunk);
	fftw_complex* fftOutput = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * audioConfig.samplesPerChunk);
	fftw_plan fftwPlan = fftw_plan_dft_1d(audioConfig.samplesPerChunk, fftInputArray, fftOutput, FFTW_FORWARD, FFTW_ESTIMATE);
	
	// Execute FFTW Plan and Convert Complex to Real
	int chunkCount = spectrogramChunks.size();
	for (int chunkNum = 0; chunkNum < chunkCount; chunkNum++) {
		for (int i = 0; i < audioConfig.samplesPerChunk; i++) {
			fftInputArray[i][0] = spectrogramChunks[chunkNum][i];
			fftInputArray[i][1] = 0;
		}

		fftw_execute(fftwPlan);

		for (int i = 0; i < audioConfig.samplesPerChunk; i++) {
			double real = fftOutput[i][0];
			double imaginary = fftOutput[i][1];

			spectrogramChunks[chunkNum][i] = sqrt(real * real + imaginary * imaginary);
		}
	}

	// Downsize Frequency Output with Average, and Take Max Value
	int valuesPerBand = audioConfig.samplesPerChunk / audioConfig.frequencyResolution;
	double maxValue = 0.0;

	for (int chunkNum = 0; chunkNum < chunkCount; chunkNum++) {
		vector<double> resultantArray;

		for (int i = 0; i < audioConfig.samplesPerChunk; i += valuesPerBand) {
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
		for (int i = 0; i < newSamplesPerChunk / 2; i++) {
			spectrogramChunks[chunkNum][i] = spectrogramChunks[chunkNum][i] / maxValue;
			spectrogramChunks[chunkNum][i] = powf(spectrogramChunks[chunkNum][i], 1.0f / audioConfig.spectrogramEmphasis);
		}

		vector<float> currentVector(spectrogramChunks[chunkNum].begin(), spectrogramChunks[chunkNum].begin() + newSamplesPerChunk / 2);
		result.push_back(currentVector);
	}

	// Return as vector<vector<float>>
	return make_pair(result, maxValue);
}
vector<int16_t> vocalSamples(const char* fullFileNameMP3, vector<vector<float>> networkOutput, audioFileConfig audioConfig) {
	// Get Max Value From Original Track (Inefficient, change before release)
	pair<vector<vector<float>>, float> fullTrackSpectrogram = spectrogramOutput(fullFileNameMP3, audioConfig);
	float maxValueSpectrogramFullTrack = fullTrackSpectrogram.second;
	
	// Recreate full spectrogram
	int networkSubOutputSize = networkOutput[0].size();
	for (int i = 0; i < networkOutput.size(); i++) {
		/*

		for (int j = 0; j < networkOutput[i].size(); j++) { // Helpful for further isolating vocals
			double hannMultiplier = 0.5 * (1 - cos(2 * 3.141 * double(j) / double(networkSubOutputSize)));
			networkOutput[i][j] = networkOutput[i][j] * pow(hannMultiplier, 0.75); // index of 0.75 widens window at top, including more frequencies
		}

		*/

		for (int j = 0; j < networkOutput[i].size(); j++) {
			networkOutput[i][j] = powf(networkOutput[i][j], audioConfig.spectrogramEmphasis);
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
	for (int i = 0; i < sampleCount - audioConfig.samplesPerChunk; i += audioConfig.samplesPerOverlap) {
		vector<double> currentChunk(doubleAudioSamples.begin() + i, doubleAudioSamples.begin() + i + audioConfig.samplesPerChunk);

		for (int j = 0; j < audioConfig.samplesPerChunk; j++) {
			double currentValue = currentChunk[j];
			currentChunk[j] = currentValue;
		}

		spectrogramChunks.push_back(currentChunk);
	}

	// FFT each chunk 1D with FFTW
	fftw_complex* fftInputArray = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * audioConfig.samplesPerChunk);
	fftw_complex* fftOutput = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * audioConfig.samplesPerChunk);
	fftw_plan fftwPlan = fftw_plan_dft_1d(audioConfig.samplesPerChunk, fftInputArray, fftOutput, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_plan fftwInversePlan = fftw_plan_dft_1d(audioConfig.samplesPerChunk, fftOutput, fftInputArray, FFTW_BACKWARD, FFTW_ESTIMATE);

	// Execute FFTW Plans
	int chunkCount = networkOutput.size();
	int frequencyResolution = networkOutput[0].size();
	int valuesPerBand = audioConfig.samplesPerChunk / audioConfig.frequencyResolution;

	vector<int16_t> resultantSamples;
	int maxNewSample = 0;

	for (int chunkNum = 0; chunkNum < chunkCount; chunkNum++) {
		for (int i = 0; i < audioConfig.samplesPerChunk; i++) {
			// Recreating full spectrogram here
			fftInputArray[i][0] = spectrogramChunks[chunkNum][i];
			fftInputArray[i][1] = 0;
		}

		fftw_execute(fftwPlan);

		for (int i = 0; i < audioConfig.samplesPerChunk; i += valuesPerBand) {
			for (int j = 0; j < valuesPerBand; j++) {
				float valueMagnitude = sqrt(fftOutput[i + j][0] * fftOutput[i + j][0] + fftOutput[i + j][1] * fftOutput[i + j][1]);
				float multiplier = sqrtf(maxValueSpectrogramFullTrack / valueMagnitude);
				
				fftOutput[i + j][0] = networkOutput[chunkNum][i / valuesPerBand] * fftOutput[i + j][0] * multiplier;
				fftOutput[i + j][1] = networkOutput[chunkNum][i / valuesPerBand] * fftOutput[i + j][1] * multiplier;
			}
		}

		fftw_execute(fftwInversePlan);

		for (int i = 0; i < audioConfig.samplesPerChunk; i++) {
			double normalisedReal = fftInputArray[i][0] / audioConfig.samplesPerChunk;
			double normalisedImag = fftInputArray[i][1] / audioConfig.samplesPerChunk;

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
	const int32_t sampleRate = 44100; // Samples Played Per Second
	int16_t channelCount = 1; 
	int32_t subChunk2Size = sampleCount * bitsPerSample * channelCount; // Size of Data (samples) chunk
	int32_t chunkSize = subChunk2Size + 32; // Size of metadata chunk
	int16_t audioFormat = 1; // Indicates No Compression used
	int32_t subChunk1Size = 16; // Initial metadata chunk size
	int32_t byteRate = sampleRate * channelCount * (bitsPerSample / 8); // Bytes Used Per Second
	int16_t blockAlign = channelCount * (bitsPerSample / 8);

	// Open File
	std::ofstream outputFile(fileName, ios::out | ios::binary);

	// Write Header Info to File
	outputFile << "RIFF";
	outputFile.write((char*)&chunkSize, sizeof(chunkSize));
	outputFile << "WAVE";

	// Add Metadata
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

	// Write Samples After Metadata
	for (int i = 0; i < sampleCount; i++) {
		int16_t currentSample = samples[i];
		outputFile.write((char*)&currentSample, sizeof(currentSample));
	}

	// Close File
	outputFile.close();
}
