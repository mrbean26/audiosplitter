# Load example chunk
from scipy.io import wavfile
sampleRate, samples = wavfile.read('Audio/scartissue.wav')

chunk = []
chunkSize = 2 ** 14

for i in range(chunkSize):
    chunk.append(samples[i])

# Hann window function
import math

def windowChunk(chunk):
    result = chunk

    chunkLength = len(chunk)
    for i in range(chunkLength):
        cosValue = math.cos(2 * math.pi * (i / (chunkLength - 1)))
        multiplier = 0.5 * (1 - cosValue)
        chunk[i] = chunk[i] * multiplier

    return result
