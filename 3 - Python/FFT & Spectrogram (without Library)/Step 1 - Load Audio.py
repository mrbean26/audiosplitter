from scipy.io import wavfile
sampleRate, samples = wavfile.read('Audio/scartissue.wav')

print(sampleRate, len(samples))
