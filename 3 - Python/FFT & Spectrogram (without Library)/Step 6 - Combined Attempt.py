# Load audio
print("1")
from scipy.io import wavfile
sampleRate, samples = wavfile.read('Audio/exampleAudio.wav')
samples = list(samples)

# Split into chunks
print("2")
samplesPerChunk = 2 ** 14
samplesPerStride = samplesPerChunk // 2
sampleCount = len(samples)

outputChunks = []
for i in range(0, sampleCount - samplesPerChunk, samplesPerStride):
    currentChunk = samples[i : i + samplesPerChunk]
    outputChunks.append(currentChunk)

# Apply window function
print("3")
import math
def getHannWindow(size):
    result = []

    for i in range(size):
        cosValue = math.cos(2 * math.pi * (i / (size - 1)))
        value = 0.5 * (1 - cosValue)
        result.append(value)

    return result

def windowChunk(chunk, window):
    result = chunk

    chunkLength = len(chunk)
    for i in range(chunkLength):
        chunk[i] = chunk[i] * window[i]

    return result

chunkCount = len(outputChunks)
hannWindow = getHannWindow(len(outputChunks[0]))
for i in range(chunkCount):
    outputChunks[i] = windowChunk(outputChunks[i], hannWindow)

# FFT each chunk
print("4")
def oddValueMultiplier(sampleLength, sampleNum):
  return math.e ** ((2.0 * math.pi * 1j * -sampleNum) / sampleLength)

# Precalculate oddvaluemultipliers as they are used more than once
oddValueMultipliers = []
for i in range(1, int(math.log(samplesPerChunk, 2)) + 1): # powers of 2 from 1 to the size of each chunk
    currentArray = []
    for j in range(2 ** i):
        currentArray.append(oddValueMultiplier(2 ** i, j))

    oddValueMultipliers.append(currentArray)

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
        twoIndex = int(math.log(sampleLength, 2))
        currentOddValueMultiplier = oddValueMultipliers[twoIndex - 1][sampleNum]

        combined[sampleNum] = frequencyEvenValues[sampleNum] + currentOddValueMultiplier * frequencyOddValues[sampleNum]
        combined[sampleNum + sampleLength // 2] = frequencyEvenValues[sampleNum] - currentOddValueMultiplier * frequencyOddValues[sampleNum]

     return combined

chunkCount = len(outputChunks)
for i in range(chunkCount):
    outputChunks[i] = FFT(outputChunks[i])
    print(i, chunkCount)

# Log magnitude, downsize frequency output with average values, and take max value
print("5")
valuesPerBand = 64
maxValue = 0

for i in range(len(outputChunks)):
    resultantArray = []

    for j in range(0, len(outputChunks[i]), valuesPerBand):
        currentAccumulative = 0

        for a in range(valuesPerBand):
            currentValue = abs(outputChunks[i][j + a])
            if currentValue > 0:
                currentValue = math.log(currentValue)

            currentAccumulative = currentAccumulative + currentValue

        currentAccumulative = currentAccumulative / valuesPerBand
        currentAccumulative = 1.5 ** currentAccumulative
        maxValue = max(maxValue, currentAccumulative)

        resultantArray.append(currentAccumulative)

    outputChunks[i] = resultantArray

# Write to image
print("7")
from PIL import Image
newImage = Image.new('RGB', (len(outputChunks), len(outputChunks[0]) // 2), color = (0, 0, 0))
imagePixels = newImage.load()

for i in range(len(outputChunks)):
    for j in range(len(outputChunks[0]) // 2):
        colourValue = int(outputChunks[i][j] / maxValue * 255)
        imagePixels[i, (len(outputChunks[0]) // 2) - 1 - j] = (colourValue, colourValue, colourValue)

newImage.save("Output Spectrograms/spectrogram.png")
