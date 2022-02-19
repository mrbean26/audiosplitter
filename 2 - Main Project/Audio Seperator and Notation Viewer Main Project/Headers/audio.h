#ifndef AUDIO_H
#define AUDIO_H

#include <vector>
#include <iostream>
using namespace std;

#include "AL/al.h"
#include "AL/alc.h"

#define SAMPLING_HZ 44100
#define M_PI 3.14159265359

struct audioFileConfig {
	int samplesPerChunk = 2048;
	int samplesPerOverlap = 2048;

	int frequencyResolution = 128;
	int chunkBorder = 4;

	int startFileIndex = 1;
	int songCount = 1;

	float spectrogramEmphasis = 2.0f; // "No emphasis" = 1.0f

	bool useOutputBinaryMask = false;
	float binaryMaskThreshold = 0.025f;

	bool useNoisePrediction = false;
};

extern int sampleRate;
extern float audioDuration;

pair<vector<vector<float>>, float> spectrogramOutput(const char* mp3Filename, audioFileConfig audioConfig);
vector<int16_t> vocalSamples(const char* fullFileNameMP3, vector<vector<float>> networkOutput, audioFileConfig audioConfig);
void writeToWAV(const char* fileName, vector<int16_t> samples);

vector<vector<float>> percentageFiltering(vector<vector<float>> inputSpectrogram, float percentageMultiplier);
vector<vector<int>> returnNoteFormat(vector<vector<float>> filteredSpectrogram);
vector<vector<int>> notesToFrets(vector<vector<int>> notes, vector<int> tunings, vector<int> maxFrets);

// Note format is vector (of instruments), then vector (of chunks), then vector (of notes)
struct instrumentConfig {
	int stringCount;
	vector<int> tunings;
	vector<int> maxFrets;
};

void saveNoteFormat(vector<pair<instrumentConfig, vector<vector<int>>>> format, const char* fileName);
vector<pair<instrumentConfig, vector<vector<int>>>> loadNoteFormat(const char* fileName);

// Audio
extern ALCdevice* device;
extern ALCcontext* context;

void startOpenAL();
class audioObject {
public:
	audioObject();

	ALuint buffer, source;
	audioObject(vector<vector<int>> unRepeatedNotes, int samplesPerChunk, int audioFileSampleRate);

	void play();
	void pause();
};

vector<ALshort> generateSinWave(float frequency, float volume, float length, int sampleRate);
vector<ALshort> accumulativeSinWave(vector<float> frequencies, vector<float> volumes, vector<float> lengths, vector<float> offsets);

vector<ALshort> notesToWave(vector<vector<int>> unRepeatedNotes, int samplesPerChunk, int audioFileSampleRate);

#endif // !AUDIO_H
