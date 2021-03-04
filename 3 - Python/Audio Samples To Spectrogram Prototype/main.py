from scipy.io import wavfile
sampleRate, samples = wavfile.read("full.wav")

maxFrequency = sampleRate / 2
samplesPerChunk = 1000
frequencyResolution = 560
acceptableZeroRange = 200

waveStartTime = 0
zeroCount = 0

maxValue = 0
frequencyVolumeTime = []
maxfreq = 0
for chunkNum in range(int(len(samples) / samplesPerChunk)):
    print(chunkNum, "/", int(len(samples) / samplesPerChunk))
    startIndex = chunkNum * samplesPerChunk

    currentFrequencyVolumeTimeChunk = []

    for count in range(frequencyResolution):
        currentFrequencyVolumeTimeChunk.append(0)

    zeroCount = 0
    waveStartTime = 0
    currentWaveMaxVolume = 0

    for sampleNum in range(startIndex + 1, startIndex + samplesPerChunk):
        lastSample = samples[sampleNum - 1]
        currentSample = samples[sampleNum]

        if zeroCount > 0:
            currentWaveMaxVolume = max(currentWaveMaxVolume, currentSample)

        if (lastSample < 0 and currentSample > 0) or (lastSample > 0 and currentSample < 0):
            zeroCount = zeroCount + 1

            if zeroCount == 1:
                waveStartTime = sampleNum / sampleRate

            if zeroCount == 3:
                waveEndTime = sampleNum / sampleRate
                waveFrequency = 1 / (waveEndTime - waveStartTime)
                maxfreq = max(maxfreq, waveFrequency)
                usedIndex = int((waveFrequency / maxFrequency) * frequencyResolution) - 1
                currentFrequencyVolumeTimeChunk[usedIndex] = currentWaveMaxVolume

                zeroCount = 0
                waveStartTime = 0
                currentWaveMaxVolume = 0

    for frequencyTime in range(frequencyResolution):
        maxValue = max(maxValue, currentFrequencyVolumeTimeChunk[frequencyTime])

    frequencyVolumeTime.append(currentFrequencyVolumeTimeChunk)

print(maxfreq)
from PIL import Image
newImage = Image.new('RGB', (len(frequencyVolumeTime), frequencyResolution), color = (255, 255, 255))
imagePixels = newImage.load()

for chunkNum in range(len(frequencyVolumeTime)):
    currentChunk = frequencyVolumeTime[chunkNum]

    for frequency in range(frequencyResolution):
        currentValue = currentChunk[frequency]
        colourValue = int((currentValue / maxValue) * 255)

        yPos = frequencyResolution - frequency - 1
        imagePixels[chunkNum, yPos] = (colourValue, 0, 0)

    #print("Done Chunk:", chunkNum, "/", len(frequencyVolumeTime))

newImage.save("spectrogram.png")
