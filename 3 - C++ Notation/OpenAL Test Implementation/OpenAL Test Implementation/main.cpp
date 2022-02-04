#include <al.h>
#include <alc.h>
#include <math.h>
#include <stdio.h>
#include <limits.h>

// Example page - https://ffainelli.github.io/openal-example/
// Page i found this code - https://github.com/hideyuki/openal-sine-wave

#define SECOND 2.0f
#define SAMPLING_HZ 44100
#define BUFFER_LENGTH (SECOND * SAMPLING_HZ)
#define SOUND_HZ 1000.0f
#define M_PI 3.14159265359

int main() {
    ALCdevice* device;
    ALCcontext* context;
    ALshort data[int(BUFFER_LENGTH) * 2];
    ALuint buffer, source;
    int i;

    // Initialization
    device = alcOpenDevice(NULL);
    context = alcCreateContext(device, NULL);
    alcMakeContextCurrent(context);
    alGenBuffers(1, &buffer);

    // Generate sine wave data
    for (i = 0; i < BUFFER_LENGTH; ++i) {
        data[i * 2] = sin(2 * M_PI * SOUND_HZ * (float(i) / float(SAMPLING_HZ))) * SHRT_MAX;
        data[i * 2 + 1] = -1 * sin(2 * M_PI * SOUND_HZ * (float(i) / float(SAMPLING_HZ))) * SHRT_MAX; // antiphase
    }

    // Output looping sine wave
    alBufferData(buffer, AL_FORMAT_STEREO16, data, sizeof(data), SAMPLING_HZ);
    alGenSources(1, &source);
    alSourcei(source, AL_BUFFER, buffer);
    alSourcei(source, AL_LOOPING, AL_FALSE);
    alSourcePlay(source);

    // Wait to exit
    printf("Press any key to exit.");
    getchar();

    // Finalization
    alSourceStop(source);
    alDeleteSources(1, &source);
    alDeleteBuffers(1, &buffer);
    alcMakeContextCurrent(NULL);
    alcDestroyContext(context);
    alcCloseDevice(device);

    return 0;
}