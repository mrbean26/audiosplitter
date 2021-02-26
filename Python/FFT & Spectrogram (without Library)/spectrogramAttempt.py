import math

# Prequisite Variables
twoIndex = 11
frequencyDownsizeScalar = 32

# Cooley-Tukey algorithm
def oddValueMultiplier(sampleLength, sampleNum):
  return math.exp((2.0 * math.pi * -sampleNum) / sampleLength)

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

# Get samples from exampleAudio.wav
from scipy.io import wavfile
sampleRate, audioSamples = wavfile.read("exampleAudio.wav")

samples = []
twoPower = 2 ** twoIndex

for i in range(math.floor(len(audioSamples) / twoPower)):
    currentArray = []

    for j in range(twoPower):
        currentArray.append(audioSamples[i * twoPower + j])
    samples.append(currentArray)

# Output FFT
for i in range(len(samples)):
    samples[i] = FFT(samples[i])

# Downsize frequency domain and get max value
outputSamples = []
maxValue = 0

for i in range(len(samples)):
    print(i, len(samples))
    currentArray = []

    for j in range(twoPower // frequencyDownsizeScalar):
        currentAccumulativeValue = 0

        for a in range(frequencyDownsizeScalar):
            currentAccumulativeValue = currentAccumulativeValue + samples[i][j * frequencyDownsizeScalar + a]

        nextValue = currentAccumulativeValue / frequencyDownsizeScalar
        maxValue = max(maxValue, nextValue)
        currentArray.append(nextValue) # Average

    outputSamples.append(currentArray)

# Change range of values from 0 to 1
for i in range(len(outputSamples)):
    for j in range(len(outputSamples[0])):
        outputSamples[i][j] = outputSamples[i][j] / maxValue

# Generate spectrogram Image
from PIL import Image
newImage = Image.new('RGB', (len(outputSamples), len(outputSamples[0])), color = (255, 255, 255))
imagePixels = newImage.load()

# Change Pixels
for i in range(len(outputSamples)):
    for j in range(len(outputSamples[0])):
        colourValue = int(outputSamples[i][j] * 255)
        imagePixels[i, len(outputSamples[0]) - 1 - j] = (colourValue, colourValue, colourValue)

# Save Image
newImage.save("outputSpectrogram.png")
