#include "Headers/audio.h"
#include <iomanip>
#include <fstream>

#define MINIMP3_IMPLEMENTATION
#include "Headers/minimp3.h"
#include "Headers/minimp3_ex.h"

#include "Headers/files.h"
#include "Headers/fftw3.h"

float audioDuration = 0.0f;
int sampleRate = 0;

// General
mp3dec_file_info_t loadAudioData(const char* mp3Filename) {
	mp3dec_t mp3Decoder;
	mp3dec_file_info_t fileInfo;

	if (mp3dec_load(&mp3Decoder, mp3Filename, &fileInfo, NULL, NULL)) {
		cout << "MP3 Load Error" << endl;
		return mp3dec_file_info_t();
	}
	
	sampleRate = fileInfo.hz;
	audioDuration = (1.0f / float(sampleRate)) * fileInfo.samples;;

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
			float networkValue = networkOutput[i][j];
			if (audioConfig.useNoisePrediction) {
				networkValue = 1.0f - networkValue;
			}

			networkOutput[i][j] = powf(networkValue, audioConfig.spectrogramEmphasis);
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

// Note format is vector (of instruments), then vector (of chunks), then vector (of notes)
void saveNoteFormat(vector<pair<instrumentConfig, vector<vector<int>>>> format, const char* fileName) {
	ofstream outputFile(fileName, ios::out | ios::binary);

	char chunkSplitCharacter = 255;
	char instrumentSplitCharacter = 254;

	int instrumentCount = format.size();
	for (int i = 0; i < instrumentCount; i++) {
		// Convert instrument data to one byte integers
		int8_t stringCount = format[i].first.stringCount;
		vector<int8_t> stringTunings(format[i].first.tunings.begin(), format[i].first.tunings.end());
		vector<int8_t> maxFrets(format[i].first.maxFrets.begin(), format[i].first.maxFrets.end());

		// Write one byte integers to file
		outputFile.write((char*)&stringCount, sizeof(stringCount));

		for (int j = 0; j < stringCount; j++) {
			outputFile.write((char*)&stringTunings[j], sizeof(stringTunings[j]));
		}
		for (int j = 0; j < stringCount; j++) {
			outputFile.write((char*)&maxFrets[j], sizeof(maxFrets[j]));
		}

		// Write note chunks to file
		int chunkCount = format[i].second.size();
		for (int j = 0; j < chunkCount; j++) {
			int noteCount = format[i].second[j].size();

			for (int k = 0; k < noteCount; k++) {
				int8_t currentNote = format[i].second[j][k];
				outputFile.write((char*)&currentNote, sizeof(currentNote));
			}

			outputFile << chunkSplitCharacter;
		}

		// split file to new instrument
		outputFile << instrumentSplitCharacter;
	}

	outputFile.close();
}
vector<pair<instrumentConfig, vector<vector<int>>>> loadNoteFormat(const char* fileName) {
	ifstream inputFile(fileName, ios::in | ios::binary);
	char currentCharacter;

	//int stringCount = (int)currentCharacter;
	//currentCharacter = inputFile.get();

	vector<pair<instrumentConfig, vector<vector<int>>>> resultantFormat;
	pair<instrumentConfig, vector<vector<int>>> currentInstrument;
	currentInstrument.first.stringCount = -1; // Used as a check if instrument config has been decoded already, starts with false

	vector<int> currentChunk;

	while (inputFile.good()) {
		currentCharacter = inputFile.get();

		if ((int)currentCharacter == -1) { // chunk split character
			currentInstrument.second.push_back(currentChunk);
			currentChunk.clear();
		}
		if ((int)currentCharacter == -2) { // Instrument split character
			resultantFormat.push_back(currentInstrument);
			currentInstrument.second.clear();

			currentInstrument.first.maxFrets.clear();
			currentInstrument.first.tunings.clear();
			currentInstrument.first.stringCount = -1;

			currentChunk.clear();
		}

		// Load notes
		if ((int)currentCharacter >= 0 && currentInstrument.first.stringCount > -1) {
			currentChunk.push_back((int)currentCharacter);
		}
		// load instrument config
		if ((int)currentCharacter >= 0 && currentInstrument.first.stringCount == -1) {
			currentInstrument.first.stringCount = (int)currentCharacter;

			// Tunings are stored first
			for (int i = 0; i < currentInstrument.first.stringCount; i++) {
				currentInstrument.first.tunings.push_back((int)inputFile.get());
			}
			// max frets are stored first
			for (int i = 0; i < currentInstrument.first.stringCount; i++) {
				currentInstrument.first.maxFrets.push_back((int)inputFile.get());
			}
		}
	}

	return resultantFormat;
}

// Audio
ALCdevice* device;
ALCcontext* context;

void startOpenAL() {
	device = alcOpenDevice(NULL);
	context = alcCreateContext(device, NULL);
	alcMakeContextCurrent(context);
}
audioObject::audioObject() {

}
audioObject::audioObject(vector<vector<int>> unRepeatedNotes, int samplesPerChunk, int audioFileSampleRate) {
	vector<ALshort> noteSineWave = notesToWave(unRepeatedNotes, samplesPerChunk, 44100);

	alGenBuffers(1, &buffer);
	alBufferData(buffer, AL_FORMAT_STEREO16, &noteSineWave[0], noteSineWave.size() * sizeof(ALshort), SAMPLING_HZ);

	alGenSources(1, &source);
	alSourcei(source, AL_BUFFER, buffer);
	//alSourcei(source, AL_LOOPING, AL_FALSE);
}

void audioObject::play() {
	alSourcePlay(source);
}
void audioObject::pause() {
	alSourcePause(source);
}

vector<ALshort> generateSinWave(float frequency, float volume, float length, int sampleRate) {
	vector<ALshort> resultantWave;

	int sampleCount = length * sampleRate;
	for (int i = 0; i < sampleCount; i++) {
		float multiplier = float(i) / float(sampleRate);

		ALshort sinValue = sin(2 * M_PI * frequency * multiplier) * SHRT_MAX * volume;
		ALshort antiphaseValue = -1 * sin(2 * M_PI * frequency * multiplier) * SHRT_MAX * volume;

		resultantWave.push_back(sinValue);
		resultantWave.push_back(antiphaseValue);
	}

	return resultantWave;
}
vector<ALshort> accumulativeSinWave(vector<float> frequencies, vector<float> volumes, vector<float> lengths, vector<float> offsets) {
	vector<vector<ALshort>> waves;

	// Get waves
	int waveCount = frequencies.size();
	for (int i = 0; i < waveCount; i++) {
		vector<ALshort> offsetWave = generateSinWave(0.0f, 0.0f, offsets[i], SAMPLING_HZ);
		vector<ALshort> currentWave = generateSinWave(frequencies[i], volumes[i], lengths[i], SAMPLING_HZ);

		offsetWave.insert(offsetWave.end(), currentWave.begin(), currentWave.end());
		waves.push_back(offsetWave);
	}

	// Average waves
	vector<ALshort> resultantWave;

	for (int i = 0; i < waveCount; i++) {
		int sampleCount = waves[i].size();
		int currentWaveSize = resultantWave.size();

		if (sampleCount > currentWaveSize) {
			resultantWave.resize(sampleCount, 0);
		}

		// Average wave
		for (int j = 0; j < sampleCount; j++) {
			if (waves[i][j] == 0) {
				continue;
			}

			int activeWaveCount = 0;
			for (int k = 0; k < waveCount; k++) {
				if (waves[k][j] != 0) {
					activeWaveCount = activeWaveCount + 1;
				}
			}

			// Sum
			resultantWave[j] = resultantWave[j] + waves[i][j] / activeWaveCount;
		}


	}

	return resultantWave;
}

vector<ALshort> notesToWave(vector<vector<int>> unRepeatedNotes, int samplesPerChunk, int audioFileSampleRate) {
	float chunkDuration = float(samplesPerChunk) / float(audioFileSampleRate);
	vector<ALshort> resultantWave;

	int chunkCount = unRepeatedNotes.size();
	for (int i = 0; i < chunkCount; i++) {
		vector<float> frequencies;
		vector<float> volumes;
		vector<float> lengths;
		vector<float> offsets;

		int noteCount = unRepeatedNotes[i].size();
		for (int j = 0; j < noteCount; j++) {
			// Note Gap Is Distance from Note A0 (frequency = 55.0f)
			float noteGapMultiplier = float(unRepeatedNotes[i][j]) / 12.0f;
			float noteFrequency = 55.0f * powf(2.0f, noteGapMultiplier);

			// Check if wave has already been created from other chunks
			if (i > 0) {
				if (find(unRepeatedNotes[i - 1].begin(), unRepeatedNotes[i - 1].end(), unRepeatedNotes[i][j]) != unRepeatedNotes[i - 1].end()) {
					break;
				}
			}

			// Generate new wave
			int chunksWithFrequency = 1;
			for (int k = i + 1; k < chunkCount; k++) {
				if (find(unRepeatedNotes[k].begin(), unRepeatedNotes[k].end(), unRepeatedNotes[i][j]) != unRepeatedNotes[k].end()) {
					chunksWithFrequency += 1;
				}
			}

			frequencies.push_back(noteFrequency);
			volumes.push_back(1.0f);
			lengths.push_back(chunkDuration * chunksWithFrequency);
			offsets.push_back(0.0f);
		}

		vector<ALshort> currentChunkWave = accumulativeSinWave(frequencies, volumes, lengths, offsets);
		resultantWave.insert(resultantWave.end(), currentChunkWave.begin(), currentChunkWave.end());
	}

	return resultantWave;
}