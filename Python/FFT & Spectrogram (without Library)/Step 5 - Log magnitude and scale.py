# Load example chunk
from scipy.io import wavfile
sampleRate, samples = wavfile.read('Audio/500hz.wav')

chunk = []
chunkSize = 2 ** 14

for i in range(chunkSize):
    chunk.append(samples[i])

# FFT - Cooley-Tukey Algorithm
import math

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

output = FFT(chunk)

# Log magnitde
maxValue = 0

for i in range(len(output)):
    for j in range(len(output[0])):
        currentValue = math.log(output[i][j])
        output[i][j] = currentValue
        maxValue = max(currentValue, maxValue)

for i in range(len(output)):
    for j in range(len(output[0])):
        output[i][j] /= maxValue
