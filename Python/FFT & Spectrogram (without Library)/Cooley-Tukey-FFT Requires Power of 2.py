import math

# Generate Wave
samples = []
timings = []

sampleRate = 512
frequency = 400
amplitude = 1
duration = 1

for i in range(duration):
    for j in range(sampleRate):
        newValue = i + (1 / sampleRate) * j
        timings.append(newValue)

for i in range(0, duration * sampleRate):
    newValue = amplitude * math.sin(math.pi * 2 * frequency * (timings[i]))
    samples.append(newValue)

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

# Output
fft_a = FFT(samples)
for i in range(len(fft_a)):
    nextVal = abs(fft_a[i].real) + abs(fft_a[i].imag)
    print(i + 1, "Hz:", nextVal)
