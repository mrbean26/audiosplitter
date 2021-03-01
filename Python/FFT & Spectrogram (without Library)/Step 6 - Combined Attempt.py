# Load audio
print("1")
from scipy.io import wavfile
sampleRate, samples = wavfile.read('Audio/exampleAudio.wav')

# Split into chunks
print("2")
samplesPerChunk = 2 ** 14
samplesPerOverlap = samplesPerChunk // 2
sampleCount = len(samples)

outputChunks = []
for i in range(0, sampleCount - samplesPerOverlap * 2, samplesPerOverlap):
    currentChunk = []

    for j in range(samplesPerChunk):
        currentChunk.append(samples[i + j])

    outputChunks.append(currentChunk)

# Apply window function
print("3")
import math
def windowChunk(chunk):
    result = chunk

    chunkLength = len(chunk)
    for i in range(chunkLength):
        cosValue = math.cos(2 * math.pi * (i / (chunkLength - 1)))
        multiplier = 0.5 * (1 - cosValue)
        chunk[i] = chunk[i] * multiplier

    return result

chunkCount = len(outputChunks)
for i in range(chunkCount):
    outputChunks[i] = windowChunk(outputChunks[i])

# FFT each chunk
print("4")
def oddValueMultiplier(sampleLength, sampleNum):
  return math.e ** ((2.0 * math.pi * 1j * -sampleNum) / sampleLength)

def FFT(signal):
  sampleLength = len(signal)
  if sampleLength == 1:
     return signal

  else:
     sampleEvenValues = []
     for i in range(0, sampleLength // 2):
         sampleEvenValues.append(signal[i * 2])

     sampleOddValues = []
     for i in range(0, sampleLength // 2):
         sampleOddValues.append(signal[i * 2 + 1])

     frequencyEvenValues = FFT(sampleEvenValues)
     frequencyOddValues = FFT(sampleOddValues)

     combined = [0] * sampleLength
     for sampleNum in range(sampleLength // 2):
        currentOddValueMultiplier = oddValueMultiplier(sampleLength, sampleNum)

        combined[sampleNum] = frequencyEvenValues[sampleNum] + currentOddValueMultiplier * frequencyOddValues[sampleNum]
        combined[sampleNum + sampleLength // 2] = frequencyEvenValues[sampleNum] - currentOddValueMultiplier * frequencyOddValues[sampleNum]

     return combined

chunkCount = len(outputChunks)
for i in range(chunkCount):
    outputChunks[i] = FFT(outputChunks[i])
    print(i, chunkCount)

# Log magnitude and change range
print("5")
for i in range(len(outputChunks)):
    for j in range(len(outputChunks[0])):
        logValue = abs(outputChunks[i][j])
        if logValue > 0:
            currentValue = math.log(logValue)
        else:
            currentValue = 0

        outputChunks[i][j] = currentValue

# Downsize frequency output
print("6")
finalChunks = []
numbersPerBand = 16
maxValue = 0

for i in range(chunkCount):
    currentArray = []

    for j in range(0, len(outputChunks[0]), numbersPerBand):
        currentAccumulative = 0

        for a in range(numbersPerBand):
            currentAccumulative += outputChunks[i][j + a]

        currentAccumulative = currentAccumulative / numbersPerBand
        currentAccumulative = 1.5 ** currentAccumulative

        maxValue = max(currentAccumulative, maxValue)
        currentArray.append(currentAccumulative)

    finalChunks.append(currentArray)

# Write to image
print("7")
from PIL import Image
newImage = Image.new('RGB', (len(finalChunks), len(finalChunks[0]) // 2), color = (0, 0, 0))
imagePixels = newImage.load()

for i in range(len(finalChunks)):
    for j in range(len(finalChunks[0]) // 2):
        colourValue = int(finalChunks[i][j] / maxValue * 255)
        imagePixels[i, (len(finalChunks[0]) // 2) - 1 - j] = (colourValue, colourValue, colourValue)

newImage.save("Output Spectrograms/Functions After Frequency Downscaling/logx.png")
