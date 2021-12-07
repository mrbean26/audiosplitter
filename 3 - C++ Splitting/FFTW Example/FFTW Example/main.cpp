#include <iostream>
using namespace std;

#include "fftw3.h"
#include "miniwav.h"

double complexToReal(double* complex) {
	return sqrt(complex[0] * complex[0] + complex[1] * complex[1]);
}

int main() {
	// Load wav samples
	const int usedSampleCount = 64;
	cout << "Audio Test 14000hz with " << usedSampleCount << " samples." << endl;

	wavData audioData = readWAVData("14000hz.wav");
	vector<double> samplesDouble(audioData.data.begin(), audioData.data.begin() + usedSampleCount);

	fftw_complex audioInputFFT[usedSampleCount];
	fftw_complex audioOutputFFT[usedSampleCount];

	for (int i = 0; i < usedSampleCount / 2; i++) {
		audioInputFFT[i][0] = samplesDouble[i];
		audioInputFFT[i][1] = 0;
	}

	fftw_plan fftwAudioPlan = fftw_plan_dft_1d(usedSampleCount, audioInputFFT, audioOutputFFT, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(fftwAudioPlan);

	fftw_destroy_plan(fftwAudioPlan);
	fftw_cleanup();

	double maxValue = 0;
	int maxValueIndex = 0;

	double secondMaxValue = 0;
	int secondMaxValueIndex = 0;

	for (int i = 0; i < usedSampleCount; i++) {
		//cout << audioOutputFFT[i][0] << " + " << audioOutputFFT[i][1] << "i" << endl;
		double totalRealValue = complexToReal(audioOutputFFT[i]);
		cout << totalRealValue << endl;

		if (totalRealValue > maxValue) {
			secondMaxValue = maxValue;
			secondMaxValueIndex = maxValueIndex;

			maxValue = totalRealValue;
			maxValueIndex = i;
		}
	}
	cout << "Max Value: " << maxValue << " at index: " << maxValueIndex << endl;
	cout << "Second Max Value: " << secondMaxValue << " at index: " << secondMaxValueIndex << endl;
	cout << endl;

	// Testing with example FFT Inputs from "http://www.sccon.ca/sccon/fft/fft3.htm"
	cout << "Test 1:1 from Link with scaling: N" << endl;
	fftw_complex inputFFT[8];
	fftw_complex outputFFT[8];

	for (int i = 0; i < 8; i++) {
		inputFFT[i][0] = 0;
		inputFFT[i][1] = 0;
	}
	inputFFT[0][0] = 1;

	fftw_plan fftwPlan = fftw_plan_dft_1d(8, inputFFT, outputFFT, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(fftwPlan);

	fftw_destroy_plan(fftwPlan);
	fftw_cleanup();

	for (int i = 0; i < 8; i++) {
		cout << outputFFT[i][0] / 8 << " + " << outputFFT[i][1] / 8 << "i" << endl;
	}
	cout << "Correct." << endl;

	// Second Test "https://stackoverflow.com/questions/6177744/what-is-correct-result-after-fft-if-input-array-is-0-1-2-3-4-5-6-7"
	cout << endl << "Test 2: {0, 1, 2, 3, 4, 5, 6, 7} with scaling: 1" << endl;
	fftw_complex inputFFTSecond[8];
	fftw_complex outputFFTSecond[8];

	for (int i = 0; i < 8; i++) {
		inputFFTSecond[i][0] = i;
		inputFFTSecond[i][1] = 0;
	}

	fftw_plan fftwPlanTwo = fftw_plan_dft_1d(8, inputFFTSecond, outputFFTSecond, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(fftwPlanTwo);

	fftw_destroy_plan(fftwPlanTwo);
	fftw_cleanup();

	for (int i = 0; i < 8; i++) {
		cout << outputFFTSecond[i][0] << " + " << outputFFTSecond[i][1] << "i" << endl;
	}
	cout << "Correct." << endl << endl;

	// Third Test
	cout << "Test 1:2 from Link with scaling: N" << endl;
	fftw_complex inputFFTThird[8];
	fftw_complex outputFFTThird[8];

	for (int i = 0; i < 8; i++) {
		inputFFTThird[i][0] = 0;
		inputFFTThird[i][1] = 0;
	}
	inputFFTThird[1][0] = 1;

	fftw_plan fftwPlanThree = fftw_plan_dft_1d(8, inputFFTThird, outputFFTThird, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(fftwPlanThree);

	fftw_destroy_plan(fftwPlanThree);
	fftw_cleanup();

	for (int i = 0; i < 8; i++) {
		cout << outputFFTThird[i][0] / 8 << " + " << outputFFTThird[i][1] / 8 << "i" << endl;
	}
	cout << "Correct." << endl;
	cout << endl;

	// Check if fftw_r2c is the same as dft_1d with all imaginary as 0
	cout << "Checking if Test 2 is equal on dft_1d and r2c" << endl;

	// complex to complex
	fftw_complex inputFFTfourth[8];
	fftw_complex outputFFTfourth[8];

	for (int i = 0; i < 8; i++) {
		inputFFTfourth[i][0] = i;
		inputFFTfourth[i][1] = 0;
	}

	fftw_plan fftwPlanFour = fftw_plan_dft_1d(8, inputFFTfourth, outputFFTfourth, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(fftwPlanFour);

	fftw_destroy_plan(fftwPlanFour);
	fftw_cleanup();

	// real to complex
	double inputFFTFifth[8];
	fftw_complex outputFFTFifth[8];

	for (int i = 0; i < 8; i++) {
		inputFFTFifth[i] = i;
	}

	fftw_plan fftwPlanFive = fftw_plan_dft_r2c_1d(8, inputFFTFifth, outputFFTFifth, FFTW_ESTIMATE);
	fftw_execute(fftwPlanFive);
	
	fftw_destroy_plan(fftwPlanFive);
	fftw_cleanup();

	// compare
	for (int i = 0; i < 8; i++) {
		cout << "C2C: " << outputFFTfourth[i][0] << " + " << outputFFTfourth[i][1] << "i,        R2C: " << outputFFTFifth[i][0] << " + " << outputFFTFifth[i][1] << "i" << endl;
	}
	

	system("pause");
	return 0;
}