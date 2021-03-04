#include <iostream>
using namespace std;

#include "fftw3.h"
#include "miniwav.h"

double complexToReal(double* complex) {
	return sqrt(complex[0] * complex[0] + complex[1] * complex[1]);
}

int main() {
	const int usedSampleCount = 64;

	// Load wav samples
	wavData audioData = readWAVData("50hz.wav");
	vector<double> samplesDouble(audioData.data.begin(), audioData.data.begin() + usedSampleCount);
	double* fftInputArray = &samplesDouble[0];

	// FFT Samples
	fftw_complex* fftOutput = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * usedSampleCount);
	fftw_plan fftwPlan = fftw_plan_dft_r2c_1d(usedSampleCount, fftInputArray, fftOutput, FFTW_ESTIMATE);
	fftw_execute(fftwPlan);

	for (int i = 0; i < usedSampleCount / 2; i++) {
		cout << complexToReal(fftOutput[i]) <<  endl;
	}

	system("pause");
	return 0;
}