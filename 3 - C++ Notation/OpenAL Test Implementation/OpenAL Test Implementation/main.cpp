#include <al.h>
#include <alc.h>

#include <iostream>
#include <vector>
using namespace std;

// Example page - https://ffainelli.github.io/openal-example/
// Page i found this code - https://github.com/hideyuki/openal-sine-wave

#define SAMPLING_HZ 44100
#define M_PI 3.14159265359

void closeOpenAL(ALuint source, ALuint buffer, ALCcontext* context, ALCdevice* device) {
    alSourceStop(source);
    alDeleteSources(1, &source);
    alDeleteBuffers(1, &buffer);
    alcMakeContextCurrent(NULL);
    alcDestroyContext(context);
    alcCloseDevice(device);
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

int main() {
    ALCdevice* device;
    ALCcontext* context;
    ALuint buffer, source;

    // Initialization
    device = alcOpenDevice(NULL);
    context = alcCreateContext(device, NULL);
    alcMakeContextCurrent(context);
    alGenBuffers(1, &buffer);

    // Generate wave
    vector<float> frequencies = { 200.0f, 800.0f };
    vector<float> volumes = { 1.0f, 0.5f };
    vector<float> lengths = { 2.0f, 1.0f };
    vector<float> offsets = { 0.0f, 1.0f };

    vector<ALshort> genData = accumulativeSinWave(frequencies, volumes, lengths, offsets);

    // Output looping sine wave
    alBufferData(buffer, AL_FORMAT_STEREO16, &genData[0], genData.size() * sizeof(ALshort), SAMPLING_HZ);
    alGenSources(1, &source);
    alSourcei(source, AL_BUFFER, buffer);
    alSourcei(source, AL_LOOPING, AL_FALSE);
    alSourcePlay(source);

    // Wait to exit
    system("pause");
    closeOpenAL(source, buffer, context, device);

    return 0;
}