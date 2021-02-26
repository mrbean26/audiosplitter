import numpy as np
import math
import matplotlib.pyplot as plt

startValue = 0
endValue = 0.5
sampleCount = 512
chosenWaveFrequency = 100

# Generate Time Intervals
timings = []
for i in range(sampleCount):
    timings.append(startValue + ((endValue - startValue) / sampleCount) * i)
samplingInterval = timings[1] - timings[0]

# Generate Sin Wave @ chosenWaveFrequency
sineWavesCombined = []
for timeValue in timings:
    sineWavesCombined.append(math.sin(math.pi * 2 * timeValue * chosenWaveFrequency))

# Generate Discrete-Fourier-Transform - https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
def dftFunc(x):
    N = len(x)

    # generate integers 0 -> N inclusive
    n = []
    for i in range(N):
        n.append(i)

    # reshape to an array of length N with subarrays of length 1
    k = []
    for i in range(N):
        k.append(n[i])

    # exp
    M = []
    for i in range(N):
        newValue = math.e ** (math.pi * k[i] * n[i] / N)
        M.append(newValue)

    # dot product M * x
    returnedValue = 0
    for i in range(N):
        returnedValue = returnedValue + M[i] * x[i]

    return returnedValue

# Generate Fast-Fourier-Transform
def fft(x):
    N = len(x)

    if N % 2 > 0:
        raise ValueError("must be a power of 2")
    elif N <= 2:
        return dftFunc(x)
    else:
        X_even_values = []
        X_odd_values = []

        for i in range(N):
            if i % 2 == 0:
                X_even_values.append(x[i])
            else:
                X_odd_values.append(x[i])

        X_even = fft(X_even_values)
        X_odd = fft(X_odd_values)

        terms = []
        for i in range(N):
            newValue = math.e ** (-2j * math.pi * i / N)
            terms.append(newValue)

        firstTerms = terms[:int(N/2)]
        firstMultiplication = []
        for i in range(len(firstTerms)):
            firstMultiplication.append(firstTerms[i] * X_odd[i] + X_even)

        secondTerms = terms[int(N/2):]
        secondMultiplication = []
        for i in range(len(secondTerms)):
            secondMultiplication.append(secondTerms[i] * X_odd + X_even)

        return firstMultiplication + secondMultiplication

ffta = fft(sineWavesCombined)

# Frequency Range for Plots
frequencyRangeArray = []
for i in range(math.floor(sampleCount / 2)):
    newValue = ((1 / samplingInterval) / sampleCount) * i
    frequencyRangeArray.append(newValue)

# Plots
plt.ylabel("Amplitude")
plt.xlabel("Frequency [Hz]")
plt.bar(frequencyRangeArray, np.abs(ffta)[:sampleCount // 2] * 1 / sampleCount, width=1.5)  # 1 / N is a normalization factor
plt.show()
