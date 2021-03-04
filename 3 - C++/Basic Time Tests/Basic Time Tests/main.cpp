#include <time.h>
#include <iostream>
#include <vector>

using namespace std;

int main() {
	vector<float> newArray(100000, 5.0f);

	clock_t startTime = clock();

	for (int i = 0; i < 10000000; i++) {
		float a = newArray[500];
	}

	clock_t retrieveTime = clock();

	//float calculated = 1.0f / (1.0f + exp(-value));

	clock_t activateTime = clock();

	//newArray[672] = calculated;

	clock_t setTime = clock();

	float retrieveDuration = float(retrieveTime - startTime) / CLOCKS_PER_SEC;
	float activateDuration = float(activateTime - retrieveTime) / CLOCKS_PER_SEC;
	float setDuration = float(setTime - activateTime) / CLOCKS_PER_SEC;
	float total = float(setTime - startTime) / CLOCKS_PER_SEC;
	cout << float(retrieveTime - startTime) / CLOCKS_PER_SEC << endl;
	cout << retrieveDuration << " " << activateDuration << " " << setDuration << endl;
	cout << retrieveDuration / total << " " << activateDuration / total << " " << setDuration / total << endl;


	system("pause");
	return 0;
}