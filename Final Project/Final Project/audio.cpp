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
int lastSeenFileSampleRate;
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

// Spectrogram functions
vector<vector<double>> spitChunks(vector<double> allSamples, int perChunk, int overlap, bool hanning) {
	vector<vector<double>> result;
	int sampleCount = allSamples.size();

	// Find hanning window values to apply to each chunk value element-wise
	vector<double> hanningWindowValues;
	if (hanning) {
		for (int i = 0; i < perChunk; i++) {
			double pi = 3.14159265358979323846;
			double currentCoefficient = double(i) / (double(perChunk) - 1);
			double cosValue = cos(2 * pi * currentCoefficient);

			double value = 0.5 * (1 - cosValue);
			hanningWindowValues.push_back(value);
		}
	}

	// go through each chunk, and multiply by hanning if neccesary
	for (int i = 0; i < sampleCount; i += overlap) {
		if (i + perChunk >= sampleCount) {
			break;
		}

		vector<double> samplesChunk(allSamples.begin() + i, allSamples.begin() + i + perChunk);
		vector<double> newChunk(perChunk);
		
		if (hanning) {
			transform(samplesChunk.begin(), samplesChunk.end(), hanningWindowValues.begin(), newChunk.begin(), multiplies<double>());
			result.push_back(newChunk);
		}
		else {
			result.push_back(samplesChunk);
		}
	}

	return result;
}
vector<vector<float>> fftChunks(vector<vector<double>> chunks, int perChunk) {
	// setup fftw plan
	fftw_complex* fftInputArray = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * perChunk);
	fftw_complex* fftOutput = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * perChunk);
	fftw_plan fftwPlan = fftw_plan_dft_1d(perChunk, fftInputArray, fftOutput, FFTW_FORWARD, FFTW_ESTIMATE);

	vector<vector<float>> result;
	int chunkCount = chunks.size();

	// execute fftw on each chunk
	for (int chunkNum = 0; chunkNum < chunkCount; chunkNum++) {
		// Set Inputs
		for (int i = 0; i < perChunk; i++) {
			fftInputArray[i][0] = chunks[chunkNum][i];
			fftInputArray[i][1] = 0;
		}

		fftw_execute(fftwPlan);

		// Set Outputs
		vector<float> newChunk;
		for (int i = 0; i < perChunk / 2; i++) { // Only take first half of spectrogram due to symmetry
			double real = fftOutput[i][0];
			double imaginary = fftOutput[i][1];

			newChunk.push_back(sqrtf(real * real + imaginary * imaginary));
		}

		result.push_back(newChunk);
	}
	
	return result;
}

vector<vector<float>> downsizeSpectrogramOutputs(vector<vector<float>> chunks, int newResolution) {
	// prequisite variables
	int currentBandCount = chunks[0].size();
	int valuesPerBand = currentBandCount / newResolution;

	int chunkCount = chunks.size();
	vector<vector<float>> result;

	// for each n values, average them out to create a single central value
	for (int i = 0; i < chunkCount; i++) {
		vector<float> currentChunk;

		for (int j = 0; j < currentBandCount; j += valuesPerBand) {
			float currentValue = 0.0f;
			
			for (int k = 0; k < valuesPerBand; k++) {
				currentValue = currentValue + chunks[i][j + k];
			}

			currentValue = currentValue / float(valuesPerBand);
			currentChunk.push_back(currentValue);
		}

		result.push_back(currentChunk);
	}

	return result;
}
vector<vector<float>> applySpectrogramEffects(vector<vector<float>> chunks, float emphasis, bool mel, int sampleRate) {
	int chunkCount = chunks.size();
	int bands = chunks[0].size();

	// 'mel perceptual weighting' is a weighting system used to represent sounds that are most audible
	// create chunkwise mel values
	vector<float> melValues;
	if (mel) {
		for (int i = 0; i < bands; i++) {
			float currentFrequency = (float(i + 1) / float(bands)) * float(sampleRate / 2.0f);
			float melScaleFactor = 2595.0f * log10f(1.0f + (currentFrequency / 700.0f));

			melValues.push_back(melScaleFactor);
		}
	}

	// apply mel scale and 'emphasis' to exaggerate highs and minimise lows
	vector<vector<float>> result;
	for (int i = 0; i < chunkCount; i++) {
		vector<float> currentChunk;

		for (int j = 0; j < bands; j++) {
			float value = chunks[i][j];

			if (mel) {
				value = value * melValues[j];
			}

			value = powf(value, 1.0f / emphasis);
			currentChunk.push_back(value);
		}

		result.push_back(currentChunk);
	}

	return result;
}

pair<vector<vector<float>>, float> normaliseSpectrogram(vector<vector<float>> spectrogram, audioFileConfig audioConfig) {
	int chunkCount = spectrogram.size();
	int bands = spectrogram[0].size();

	float maxValue = 0.0f;
	for (int i = 0; i < chunkCount; i++) {
		for (int j = 0; j < bands; j++) {
			maxValue = max(maxValue, spectrogram[i][j]);
		}
	}

	for (int i = 0; i < chunkCount; i++) {
		for (int j = 0; j < bands; j++) {
			spectrogram[i][j] = spectrogram[i][j] / maxValue;

			if (audioConfig.useNoisePrediction) {
				spectrogram[i][j] = 1.0f - spectrogram[i][j];
			}
		}
	}

	return make_pair(spectrogram, maxValue);
}

pair<vector<vector<float>>, float> spectrogramOutput(const char* mp3Filename, audioFileConfig audioConfig) {
	mp3dec_file_info_t audioData = loadAudioData(mp3Filename);
	int sampleRate = audioData.hz;
	lastSeenFileSampleRate = sampleRate;

	vector<int> audioSamples = loadAudioSamples(audioData.buffer, audioData.samples, audioData.channels);
	delete[] audioData.buffer; 
	
	vector<double> doubleAudioSamples(audioSamples.begin(), audioSamples.end());
	vector<vector<double>> spectrogramChunks = spitChunks(doubleAudioSamples, audioConfig.samplesPerChunk, audioConfig.samplesPerOverlap, true);
	
	vector<vector<float>> fftOutputs = fftChunks(spectrogramChunks, audioConfig.samplesPerChunk);
	vector<vector<float>> downsizedOutputs = downsizeSpectrogramOutputs(fftOutputs, audioConfig.frequencyResolution / 2);
	
	vector<vector<float>> effectsSpectrogram = applySpectrogramEffects(downsizedOutputs, audioConfig.spectrogramEmphasis, audioConfig.useMelScale, sampleRate);
	pair<vector<vector<float>>, float> normalisedSpectrogram = normaliseSpectrogram(effectsSpectrogram, audioConfig);

	return make_pair(normalisedSpectrogram.first, normalisedSpectrogram.second);
}
pair<vector<vector<float>>, float> spectrogramOutput(vector<vector<double>> samplesChunks, audioFileConfig audioConfig, int sampleRate) {
	vector<vector<float>> fftOutputs = fftChunks(samplesChunks, audioConfig.samplesPerChunk);
	vector<vector<float>> downsizedOutputs = downsizeSpectrogramOutputs(fftOutputs, audioConfig.frequencyResolution / 2);

	vector<vector<float>> effectsSpectrogram = applySpectrogramEffects(downsizedOutputs, audioConfig.spectrogramEmphasis, audioConfig.useMelScale, sampleRate);
	pair<vector<vector<float>>, float> normalisedSpectrogram = normaliseSpectrogram(effectsSpectrogram, audioConfig);

	return make_pair(normalisedSpectrogram.first, normalisedSpectrogram.second);
}

// vocal output functions
pair<vector<int16_t>, int> networkOutputToSamples(vector<vector<float>> networkOutput, vector<vector<double>> spectrogramChunks, audioFileConfig audioConfig) {
	// Prequisite Variables
	fftw_complex* fftInputArray = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * audioConfig.samplesPerChunk);
	fftw_complex* fftOutput = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * audioConfig.samplesPerChunk);
	fftw_plan fftwPlan = fftw_plan_dft_1d(audioConfig.samplesPerChunk, fftInputArray, fftOutput, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_plan fftwInversePlan = fftw_plan_dft_1d(audioConfig.samplesPerChunk, fftOutput, fftInputArray, FFTW_BACKWARD, FFTW_ESTIMATE);
	
	int chunkCount = networkOutput.size();
	int frequencyResolution = networkOutput[0].size();
	int valuesPerBand = audioConfig.samplesPerChunk / audioConfig.frequencyResolution;
	
	// FFT, Multiply then IFFT
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
				fftOutput[i + j][0] = networkOutput[chunkNum][i / valuesPerBand] * fftOutput[i + j][0];
				fftOutput[i + j][1] = networkOutput[chunkNum][i / valuesPerBand] * fftOutput[i + j][1];
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

	return make_pair(resultantSamples, maxNewSample);
}
vector<int16_t> vocalSamples(const char* fullFileNameMP3, vector<vector<float>> networkOutput, audioFileConfig audioConfig) {
	// Get Max Value From Original Track (Inefficient, change before release)
	audioConfig.samplesPerOverlap = audioConfig.samplesPerChunk;

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

	// load back in original track
	mp3dec_file_info_t audioData = loadAudioData(fullFileNameMP3);
	vector<int> audioSamples = loadAudioSamples(audioData.buffer, audioData.samples, audioData.channels);

	int maxInitialSample = 0;
	for (int i = 0; i < audioSamples.size(); i++) {
		maxInitialSample = max(maxInitialSample, abs(audioSamples[i]));
	}

	vector<double> doubleAudioSamples(audioSamples.begin(), audioSamples.end());
	vector<vector<double>> spectrogramChunks = spitChunks(doubleAudioSamples, audioConfig.samplesPerChunk, audioConfig.samplesPerOverlap, false);
	pair<vector<int16_t>, int> resultantSamples = networkOutputToSamples(networkOutput, spectrogramChunks, audioConfig);

	// amplify to appropirate value
	for (int i = 0; i < resultantSamples.first.size(); i++) {
		resultantSamples.first[i] = resultantSamples.first[i] * (double(maxInitialSample) / double(resultantSamples.second));
	}

	return resultantSamples.first;
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
void writeToWAV(const char* fileName, vector<int16_t> samples) {
	int sampleCount = samples.size();

	// neccesary wave metadata
	int16_t bitsPerSample = 16;
	const int32_t sampleRate = lastSeenFileSampleRate; // Samples Played Per Second
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
