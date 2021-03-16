from scipy.io import wavfile
import numpy
import math

# Variables
samplesPerChunk = 8192
frequencyResolution = 2048
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

# Downsize vocals
valuesPerBand = samplesPerChunk // frequencyResolution

for i in range(chunkCount):
    outptutChunk = []

    for j in range(0, samplesPerChunk, valuesPerBand):
        accumulative = 0

        for k in range(valuesPerBand):
            accumulative = accumulative + vocalTrackChunks[i][j + k]

        accumulative = accumulative / valuesPerBand
        outptutChunk.append(accumulative)

    vocalTrackChunks[i] = outptutChunk

# Find max value in vocal track
maxValue = 0

for i in range(chunkCount):
    for j in range(frequencyResolution):
        maxValue = max(maxValue, abs(vocalTrackChunks[i][j]))

# Find max value in full track
maxValueFull = 0
for i in range(chunkCount):
    for j in range(samplesPerChunk):
        maxValueFull = max(maxValueFull, abs(fullTrackChunks[i][j]))

# Convert vocal track to 0s or 1s
vocalThresholdValue = vocalThreshold * maxValue

for i in range(chunkCount):
    for j in range(frequencyResolution):
        vocalTrackChunks[i][j] = abs(vocalTrackChunks[i][j]) / maxValue

# Remove complex values when corresponding vocal is 0
for i in range(chunkCount):
    for j in range(frequencyResolution):
        for k in range(valuesPerBand):
            fullTrackChunks[i][j * valuesPerBand + k] = fullTrackChunks[i][j * valuesPerBand + k] * vocalTrackChunks[i][j] * (maxValueFull / maxValue)

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
