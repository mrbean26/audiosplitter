import numpy as np
from scipy.io import wavfile

rate, data = wavfile.read('Audio/exampleAudio.wav')

def spectrogram(samples, sample_rate, stride_ms = 10.0,
                          window_ms = 20.0, max_freq = None, eps = 1e-14):

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples,
                                          shape = nshape, strides = nstrides)

    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2

    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale

    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    # Compute spectrogram feature
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[:ind, :] + eps)

    return specgram

returned = spectrogram(data, rate, max_freq = rate / 2)

maxValue = 0
for i in range(len(returned)):
    for j in range(len(returned[0])):
        maxValue = max(maxValue, returned[i][j])

# Create image
from PIL import Image
newImage = Image.new('RGB', (len(returned[0]), len(returned)), color = (0, 0, 0))
imagePixels = newImage.load()

for i in range(len(returned)):
    for j in range(len(returned[0])):
        colourValue = int(returned[i][j] / maxValue * 255)
        imagePixels[j, len(returned) - 1 - i] = (colourValue, colourValue, colourValue)

newImage.save("Output Spectrograms/spectrogram.png")
