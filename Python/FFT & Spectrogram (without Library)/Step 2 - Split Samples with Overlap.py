from scipy.io import wavfile
sampleRate, samples = wavfile.read('Audio/scartissue.wav')

# Samples per chunk being a power of 2 is ideal
samplesPerChunk = 2 ** 14
samplesPerOverlap = samplesPerChunk // 2
sampleCount = len(samples)

outputChunks = []
for i in range(0, sampleCount - samplesPerOverlap * 2, samplesPerOverlap):
    currentChunk = []

    for j in range(samplesPerChunk):
        currentChunk.append(samples[i + j])

    outputChunks.append(currentChunk)

# Check
print("Chunk Length:", len(outputChunks[0]))
print("Chunk Count:", len(outputChunks))
