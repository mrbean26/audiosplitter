from scipy.io import wavfile
import numpy
import math

# Variables
samplesPerChunk = 8192
frequencyResolution = 128
vocalThreshold = 0.005

# Read Files
sampleRate, fullTrackSamples = wavfile.read("input.wav")
sampleRateTwo, vocalTrackSamples = wavfile.read("output.wav")

# Split into chunks
fullTrackChunks = []
vocalTrackChunks = []

for i in range(0, len(fullTrackSamples) - samplesPerChunk, samplesPerChunk):
    currentChunk = []

    for j in range(samplesPerChunk):
        currentChunk.append(fullTrackSamples[i + j])

    fullTrackChunks.append(currentChunk)

for i in range(0, len(vocalTrackSamples) - samplesPerChunk, samplesPerChunk):
    currentChunk = []

    for j in range(samplesPerChunk):
        currentChunk.append(vocalTrackSamples[i + j])

    vocalTrackChunks.append(currentChunk)

# FFT Each chunk
chunkCount = len(fullTrackChunks)

for i in range(chunkCount):
    fullTrackChunks[i] = numpy.fft.fft(fullTrackChunks[i])
    vocalTrackChunks[i] = numpy.fft.fft(vocalTrackChunks[i])

# Find max value in vocals
maxValue = 0

for i in range(chunkCount):
    for j in range(samplesPerChunk):
        maxValue = max(maxValue, abs(vocalTrackChunks[i][j]))

# Convert vocal track to 0s or 1s
vocalThresholdValue = vocalThreshold * maxValue

for i in range(chunkCount):
    for j in range(samplesPerChunk):
        value = abs(vocalTrackChunks[i][j])
        vocalTrackChunks[i][j] = abs(vocalTrackChunks[i][j]) / maxValue

# Remove complex values when corresponding vocal is 0
for i in range(chunkCount):
    for j in range(samplesPerChunk):
        fullTrackChunks[i][j] = fullTrackChunks[i][j] * vocalTrackChunks[i][j]

# IFFT

for i in range(chunkCount):
    fullTrackChunks[i] = numpy.fft.ifft(fullTrackChunks[i])


# Combine chunks
outputTrackSamples = []

for i in range(chunkCount):
    for j in range(samplesPerChunk):
        value = fullTrackChunks[i][j].real

        outputTrackSamples.append(value)


# Write to file
outputArray = numpy.asarray(outputTrackSamples, dtype = numpy.int16)
wavfile.write("outputAudio.wav", sampleRate, outputArray)
