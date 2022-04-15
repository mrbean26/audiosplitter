#include "Headers/audio.h"
#include <iomanip>
#include <fstream>

#define MINIMP3_IMPLEMENTATION
#include "Headers/minimp3.h"
#include "Headers/minimp3_ex.h"

#include "Headers/stb_image_write.h"
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
		// Convert Stereo to Mono by averaging left and right side samples (mean)

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

	// Generate hanning window values to make middle values most "important", count = samplesPerChunk
	vector<double> hanningWindowValues;
	for (int i = 0; i < audioConfig.samplesPerChunk; i++) {
		double pi = 3.14159265358979323846;
		double currentCoefficient = double(i) / (double(audioConfig.samplesPerChunk) - 1);
		double cosValue = cos(2 * pi * currentCoefficient);

		double value = 0.5 * (1 - cosValue);
		hanningWindowValues.push_back(value);
	}

	// Split into chunks (FFT -> STFT) & apply hanning window function
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

		// Take magnitude of each complex number
		for (int i = 0; i < audioConfig.samplesPerChunk; i++) {
			double real = fftOutput[i][0];
			double imaginary = fftOutput[i][1];

			spectrogramChunks[chunkNum][i] = sqrtf(real * real + imaginary * imaginary);
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

	// Change Values from Range 0 to 1 and remove bottom of half of spectrogram as they are just reflections of each other - unneccesarry network complexity
	vector<vector<float>> result;
	int newSamplesPerChunk = spectrogramChunks[0].size();

	for (int chunkNum = 0; chunkNum < chunkCount; chunkNum++) {
		for (int i = 0; i < newSamplesPerChunk / 2; i++) {
			spectrogramChunks[chunkNum][i] = spectrogramChunks[chunkNum][i] / maxValue;
			spectrogramChunks[chunkNum][i] = powf(spectrogramChunks[chunkNum][i], 1.0f / audioConfig.spectrogramEmphasis);
		}

		// remove bottom half
		vector<float> currentVector(spectrogramChunks[chunkNum].begin(), spectrogramChunks[chunkNum].begin() + newSamplesPerChunk / 2);
		
		// mel scale
		if (audioConfig.useMelScale) {
			for (int i = 0; i < newSamplesPerChunk / 2; i++) {
				float currentFrequency = (float(i) / float(newSamplesPerChunk / 2)) * 22050.0f;
				currentVector[i] = currentVector[i] * (2595.0f * log10f(1.0f + (currentFrequency / 700.0f))); // Mel scale
			}
		}

		result.push_back(currentVector);
	}

	// Return as vector<vector<float>>
	return make_pair(result, maxValue);
}
void writeSpectrogramToImage(vector<vector<float>> spectrogram, const char* fileNameJPG) {
	int width = spectrogram.size();
	int height = spectrogram[0].size();
	int channelCount = 3;

	unsigned char* data = new unsigned char[width * height * channelCount];
	int index = 0;

	for (int y = height - 1; y >= 0; y--) {
		for (int x = 0; x < width; x++) {
			int intColour = int(255.0 * spectrogram[x][y]);

			data[index++] = intColour;
			data[index++] = intColour;
			data[index++] = intColour;
		}
	}

	stbi_write_jpg(fileNameJPG, width, height, channelCount, data, width * sizeof(int));
}

vector<int16_t> vocalSamples(const char* fullFileNameMP3, vector<vector<float>> networkOutput, audioFileConfig audioConfig) {
	// Get Max Value From Original Track (Inefficient, change before release)
	audioConfig.samplesPerOverlap = audioConfig.samplesPerChunk;
	pair<vector<vector<float>>, float> fullTrackSpectrogram = spectrogramOutput(fullFileNameMP3, audioConfig);
	float maxValueSpectrogramFullTrack = fullTrackSpectrogram.second;

	// Recreate full spectrogram
	int networkSubOutputSize = networkOutput[0].size();
	for (int i = 0; i < networkOutput.size(); i++) {
		for (int j = 0; j < networkOutput[i].size(); j++) {
			float networkValue = networkOutput[i][j];
			if (audioConfig.useNoisePrediction) {
				networkValue = 1.0f - networkValue;
			}

			if (!audioConfig.useOutputBinaryMask) {
				networkOutput[i][j] = powf(networkValue, audioConfig.spectrogramEmphasis);
			}
		}

		vector<float> currentChunk = networkOutput[i];
		currentChunk.insert(currentChunk.end(), networkOutput[i].begin(), networkOutput[i].end()); // Add bottom half back in

		networkOutput[i] = currentChunk;
	}

	// IFFT Total
	// load back in original track
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
	for (int i = 0; i < sampleCount - audioConfig.samplesPerChunk; i += audioConfig.samplesPerChunk) {
		vector<double> currentChunk(doubleAudioSamples.begin() + i, doubleAudioSamples.begin() + i + audioConfig.samplesPerChunk);
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
			// Multiply network output vocal mask by full track values
			for (int j = 0; j < valuesPerBand; j++) {
				float valueMagnitude = sqrt(fftOutput[i + j][0] * fftOutput[i + j][0] + fftOutput[i + j][1] * fftOutput[i + j][1]);

				//float multiplier = sqrtf(maxValueSpectrogramFullTrack / valueMagnitude); // to do with amplification
				float multiplier = 1.0f;

				fftOutput[i + j][0] = networkOutput[chunkNum][i / valuesPerBand] * fftOutput[i + j][0] * multiplier;
				fftOutput[i + j][1] = networkOutput[chunkNum][i / valuesPerBand] * fftOutput[i + j][1] * multiplier;
			}
		}

		fftw_execute(fftwInversePlan);
		// Convert inversed fft back into samples
		for (int i = 0; i < audioConfig.samplesPerChunk; i++) {
			double normalisedReal = fftInputArray[i][0] / audioConfig.samplesPerChunk;
			double normalisedImag = fftInputArray[i][1] / audioConfig.samplesPerChunk;

			int16_t currentSample = normalisedReal;
			maxNewSample = max(maxNewSample, currentSample);
			resultantSamples.push_back(currentSample);
		}
	}

	// cleanup fftw data
	fftw_destroy_plan(fftwPlan);
	fftw_destroy_plan(fftwInversePlan);
	fftw_free(fftInputArray);
	fftw_free(fftOutput);

	// amplify to appropirate value
	for (int i = 0; i < resultantSamples.size(); i++) {
		resultantSamples[i] = resultantSamples[i] * (double(maxInitialSample) / double(maxNewSample));
	}

	return resultantSamples;
}
void writeToWAV(const char* fileName, vector<int16_t> samples) {
	int sampleCount = samples.size();

	// neccesary wave metadata
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
