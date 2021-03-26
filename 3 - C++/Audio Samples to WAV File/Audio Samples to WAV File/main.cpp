#include <iostream>
#include <fstream>
using namespace std;

int main() {
	// Declare Data
	const int duration = 2;
	int16_t bitsPerSample = 16;

	const int32_t sampleRate = 44100;
	int16_t channelCount = 1;
	int32_t subChunk2Size = sampleRate * duration * bitsPerSample * channelCount;
	int32_t chunkSize = subChunk2Size + 32;
	int16_t audioFormat = 1;
	int32_t subChunk1Size = 16;
	int32_t byteRate = sampleRate * channelCount * (bitsPerSample / 8);
	int16_t blockAlign = channelCount * (bitsPerSample / 8);
	
	// Generate Samples
	int frequency = 4000;
	double pi = 3.141;

	int16_t outputSampleWave[sampleRate * duration];
	for (int i = 0; i < sampleRate * duration; i++) {
		double sample = sin(2 * pi * i * (double(frequency) / double(sampleRate)));
		int16_t finalSample = sample * 32768.0;
		outputSampleWave[i] = finalSample;
	}

	// Open File
	std::ofstream outputFile("outputAudio.wav", ios::out | ios::binary);

	// Write Header Info to File
	outputFile << "RIFF";
	outputFile.write((char*)&chunkSize, sizeof(chunkSize));
	outputFile << "WAVE";

	outputFile << "fmt ";
	outputFile.write((char*)&subChunk1Size, sizeof(subChunk1Size));
	outputFile.write((char*)&audioFormat, sizeof(audioFormat));
	outputFile.write((char*)&channelCount, sizeof(channelCount));
	outputFile.write((char*)&sampleRate, sizeof(sampleRate));
	outputFile.write((char*)&byteRate, sizeof(byteRate));
	outputFile.write((char*)&blockAlign, sizeof(blockAlign));
	outputFile.write((char*)&bitsPerSample, sizeof(bitsPerSample));

	// Data Chunk
	outputFile << "data";
	outputFile.write((char*)&subChunk2Size, sizeof(subChunk2Size));
	
	for (int i = 0; i < sampleRate * duration; i++) {
		int16_t currentSample = outputSampleWave[i];
		outputFile.write((char*)&currentSample, sizeof(currentSample));
	}
	// Close
	outputFile.close();

	system("pause");
	return 0;
}