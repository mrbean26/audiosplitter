#pragma once
#ifndef MINI_WAV
#define MINI_WAV

#include <vector>
#include <fstream>
#include <string>
using namespace std;

struct wavData {
	int channelCount, sampleRate, byteRate, bytesPerSample, bitsPerChannel;
	vector<int> data;
};

int binaryToInteger(string binary);
int signedBinaryToInteger(string binary);
wavData readWAVData(const char* fileName);
string charToBinaryString(char c);
vector<char> readFileIntoBytes(const char* fileName);

wavData readWAVData(const char* fileName) {
	vector<char> fileBytes = readFileIntoBytes(fileName);
	wavData result = wavData();

	// headerInformation
	result.channelCount = binaryToInteger(charToBinaryString(fileBytes[23]) + charToBinaryString(fileBytes[22]));

	string sampleRateString = charToBinaryString(fileBytes[27]) + charToBinaryString(fileBytes[26]) + charToBinaryString(fileBytes[25]) + charToBinaryString(fileBytes[24]);
	result.sampleRate = binaryToInteger(sampleRateString);
	
	string byteRateString = charToBinaryString(fileBytes[31]) + charToBinaryString(fileBytes[30]) + charToBinaryString(fileBytes[29]) + charToBinaryString(fileBytes[28]);
	result.byteRate = binaryToInteger(byteRateString);

	result.bytesPerSample = binaryToInteger(charToBinaryString(fileBytes[32]));
	result.bitsPerChannel = binaryToInteger(charToBinaryString(fileBytes[34]));

	string dataSizeBytesString = charToBinaryString(fileBytes[43]) + charToBinaryString(fileBytes[42]) + charToBinaryString(fileBytes[41]) + charToBinaryString(fileBytes[40]);
	int dataSizeBytes = binaryToInteger(dataSizeBytesString);
	int sampleCount = (dataSizeBytes / result.channelCount) / (result.bitsPerChannel / 8);

	for (int i = 0; i < dataSizeBytes / (result.bitsPerChannel / 8); i++) {
		string currentChannelSample = "";
		for (int c = (result.bitsPerChannel / 8) - 1; c >= 0; c--) {
			int index = i * (result.bitsPerChannel / 8) + c;
			currentChannelSample += charToBinaryString(fileBytes[44 + index]);
		}
		result.data.push_back(signedBinaryToInteger(currentChannelSample));
	}

	return result;
}

vector<char> readFileIntoBytes(const char* fileName) {
	ifstream file(fileName, ios::binary | ios::in);
	char currentCharacter;

	vector<char> result;
	while (file.get(currentCharacter)) {
		result.push_back(currentCharacter);
	}
	return result;
}

int binaryToInteger(string binary) {
	return stoi(binary, nullptr, 2);
}

int signedBinaryToInteger(string binary) {
	vector<char> charResult;
	copy(binary.begin(), binary.end(), back_inserter(charResult));

	int result = 0;
	int length = binary.length();

	for (int i = length - 1; i >= 0; i--) {
		if (charResult[i] == '1') {
			if (i == 0) {
				result -= pow(2, (length - 1));
				continue;
			}
			result += pow(2, length - 1 - i);
		}
	}

	return result;
}

string charToBinaryString(char c) {
	string result = "";
	for (int i = 7; i >= 0; i--) {
		result += to_string((c >> i) & 1);
	}

	return result;
}


#endif // !MINI_WAV
