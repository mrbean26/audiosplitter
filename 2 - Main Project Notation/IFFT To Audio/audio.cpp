#include "Headers/audio.h"
#include <iomanip>
#include <fstream>

#define MINIMP3_IMPLEMENTATION
#include "Headers/minimp3.h"
#include "Headers/minimp3_ex.h"

int sampleRate = 0;
float audioDuration = 0.0f;

mp3dec_file_info_t loadAudioData(const char* mp3Filename) {
	mp3dec_t mp3Decoder;
	mp3dec_file_info_t fileInfo;

	if (mp3dec_load(&mp3Decoder, mp3Filename, &fileInfo, NULL, NULL)) {
		cout << "MP3 Load Error" << endl;
		return mp3dec_file_info_t();
	}
	
	sampleRate = fileInfo.hz;
	audioDuration = (1.0f / float(sampleRate)) * fileInfo.samples;

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
pair<vector<vector<float>>, float> spectrogramOutput(const char* mp3Filename, int samplesPerChunk, int samplesPerStride, int frequencyResolution) {
	mp3dec_file_info_t audioData = loadAudioData(mp3Filename);
	vector<int> audioSamples = loadAudioSamples(audioData.buffer, audioData.samples, audioData.channels);

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
			
			/* BINARY MASK
			if (spectrogramChunks[chunkNum][i] > 0.025f) {
				spectrogramChunks[chunkNum][i] = 1.0f;
			}
			else {
				spectrogramChunks[chunkNum][i] = 0.0f;
			}
			*/
		}

		vector<float> currentVector(spectrogramChunks[chunkNum].begin(), spectrogramChunks[chunkNum].begin() + newSamplesPerChunk / 2);
		result.push_back(currentVector);
	}

	// Return as vector<vector<float>>
	return make_pair(result, maxValue);
}

float randomFloat(float min, float max) {
	float random = float(rand()) / float(RAND_MAX);
	float diff = max - min;
	float r = random * diff;
	return min + r;
}
pair<vector<vector<float>>, float> addSpectrogramError(pair<vector<vector<float>>, float> original, float error) {
	pair<vector<vector<float>>, float> result = original;
	float minError = -(error / 2.0f);
	float maxError = error / 2.0f;

	int columnCount = result.first.size();
	int rowCount = result.first[0].size();

	for (int i = 0; i < columnCount; i++) {
		for (int j = 0; j < rowCount; j++) {
			float addedError = randomFloat(minError, maxError);
			result.first[i][j] = result.first[i][j] + addedError;
		}
	}

	return result;
}

vector<vector<float>> percentageFiltering(vector<vector<float>> inputSpectrogram, float percentageMultiplier) {
	vector<vector<float>> result;

	for (int i = 0; i < inputSpectrogram.size(); i++) {
		vector<float> resultantChunk;

		float maxValue = 0.0f;
		for (int j = 0; j < inputSpectrogram[0].size(); j++) {
			maxValue = max(maxValue, inputSpectrogram[i][j]);
		}

		float threshold = maxValue * percentageMultiplier;
		for (int j = 0; j < inputSpectrogram[0].size(); j++) {
			if (inputSpectrogram[i][j] > threshold) {
				resultantChunk.push_back(inputSpectrogram[i][j]);
			}
			else {
				resultantChunk.push_back(0.0f);
			}
		}

		result.push_back(resultantChunk);
	}
	return result;
}
vector<vector<int>> returnNoteFormat(vector<vector<float>> filteredSpectrogram) {
	vector<vector<int>> resultantMIDIFormat; // Each vector<int> is a time frame and each int is an integer distance "note"

	int timeframes = filteredSpectrogram.size();
	int frequencies = filteredSpectrogram[0].size();

	for (int i = 0; i < timeframes; i++) {
		vector<int> currentTimeFrame;

		for (int j = 0; j < frequencies; j++) {
			// Check if note is acceptable volume
			if (filteredSpectrogram[i][j] > 0.0f) {
				float predictedFrequency = (float(j) / float(frequencies)) * (sampleRate / 2);
				int noteGap = log2f(predictedFrequency / 55.0f) * 12;
				// Note Gap Is Distance from Note A0 (frequency = 55.0f)

				if (noteGap < 0) {
					continue;
				}

				currentTimeFrame.push_back(noteGap);
			}
		}
		resultantMIDIFormat.push_back(currentTimeFrame);
	}
	return resultantMIDIFormat;
}
vector<vector<int>> notesToFrets(vector<vector<int>> notes, vector<int> tunings, vector<int> maxFrets) {
	vector<vector<int>> result;
	int stringCount = tunings.size();
	int frameCount = notes.size();

	for (int i = 0; i < frameCount; i++) {
		vector<int> currentFrame(stringCount);
		std::fill(currentFrame.begin(), currentFrame.end(), -1); // Fill with -1's, meaning no fret

		// Loop over all notes in frame
		int noteCount = notes[i].size();
		for (int j = 0; j < noteCount; j++) {
			int lowestDifference = INT_MAX;
			int chosenStringIndex = -1;

			for (int k = 0; k < stringCount; k++) {
				int currentDifference = notes[i][j] - tunings[k]; // Find Fret Number of Note on this String

				// Find lowest fret number
				if (currentDifference < lowestDifference && currentDifference >= 0) {
					if (currentDifference <= tunings[k]) {
						chosenStringIndex = k;
						lowestDifference = currentDifference;
					}
				}
			}

			// Assign note to a fret on chosen string
			for (int k = chosenStringIndex; k >= 0; k--) {
				if (currentFrame[k] == -1) {
					currentFrame[k] = notes[i][j] - tunings[k];
					break;
				}

				// If the string is already occupied by a note, push note up with lowest fret
				int newNoteFret = notes[i][j] - tunings[k];
				int currentNoteFret = currentFrame[k];

				if (currentNoteFret < newNoteFret) {
					notes[i][j] = currentNoteFret;
				}
			}
		}
		result.push_back(currentFrame);
	}
	return result;
}

void saveNoteFormat(vector<vector<int>> format, int stringCount, const char* fileName) {
	ofstream outputFile(fileName, ios::out | ios::binary);

	int8_t oneByteStringCount = stringCount;
	outputFile.write((char*)&oneByteStringCount, sizeof(oneByteStringCount));

	char splitCharacter = 255;
	int chunkCount = format.size();

	for (int i = 0; i < chunkCount; i++) {
		int noteCount = format[i].size();

		for (int j = 0; j < noteCount; j++) {
			int8_t currentNote = format[i][j];
			outputFile.write((char*)&currentNote, sizeof(currentNote));
		}

		outputFile << splitCharacter;
	}

	outputFile.close();
}
vector<vector<int>> loadNoteFormat(const char* fileName) {
	ifstream inputFile(fileName, ios::in | ios::binary);
	
	char currentCharacter = inputFile.get();
	int stringCount = (int)currentCharacter;
	currentCharacter = inputFile.get();

	vector<vector<int>> resultantFormat;
	vector<int> currentChunk;

	while (inputFile.good()) {
		if ((int)currentCharacter == -1) {
			resultantFormat.push_back(currentChunk);
			currentChunk.clear();
		}
		else {
			currentChunk.push_back((int)currentCharacter);
		}

		currentCharacter = inputFile.get();
	}

	return resultantFormat;
}