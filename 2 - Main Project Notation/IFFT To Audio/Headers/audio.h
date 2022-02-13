#ifndef AUDIO_H
#define AUDIO_H

#include <vector>
#include <iostream>
using namespace std;

#include "Headers/fftw3.h"

#define SAMPLING_HZ 44100
#define M_PI 3.14159265359

#include "AL/al.h"
#include "AL/alc.h"

extern float audioDuration;
pair<vector<vector<float>>, float> spectrogramOutput(const char* mp3Filename, int samplesPerChunk, int samplesPerStride, int frequencyResolution);
pair<vector<vector<float>>, float> addSpectrogramError(pair<vector<vector<float>>, float> original, float error);

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
	ALuint buffer, source;
	audioObject(vector<vector<int>> unRepeatedNotes, int samplesPerChunk, int audioFileSampleRate);

	void play();
	void pause();
};

vector<ALshort> generateSinWave(float frequency, float volume, float length, int sampleRate);
vector<ALshort> accumulativeSinWave(vector<float> frequencies, vector<float> volumes, vector<float> lengths, vector<float> offsets);

vector<ALshort> notesToWave(vector<vector<int>> unRepeatedNotes, int samplesPerChunk, int audioFileSampleRate);

#endif // !AUDIO_H
