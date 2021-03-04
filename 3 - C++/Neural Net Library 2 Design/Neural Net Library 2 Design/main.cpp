#include <iostream>
#include <time.h>
using namespace std;

#include "Headers/NeuralNetwork.h"

int main() {
	srand(time(NULL));

	NeuralNetwork newNetwork({ 2, 2, 1 }, { 1, 1, 0 }, "tanh");
	vector<vector<float>> trainInputs = { {0.0f, 0.0f}, {1.0f, 0.0f} , {0.0f, 1.0f}, {1.0f, 1.0f} };
	vector<vector<float>> trainOutputs = { {0.0f}, {1.0f}, {1.0f}, {0.0f} };

	newNetwork.train(100000, 0.1f, 0.05f, trainInputs, trainOutputs);
	newNetwork.runTests(trainInputs);

	system("pause");
	
	return 0;
}