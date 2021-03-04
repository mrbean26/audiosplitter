#include <vector>
#include <iostream>
#include <chrono>

#include <algorithm>
#include <functional>
#include <string>

#include <valarray>
using namespace std;

string addVectorsTimeTest(int count, float values) {
	auto start = chrono::high_resolution_clock::now();

	valarray<float> firstVector(values, count);
	valarray<float> secondVector(values, count);
	valarray<float> resultVector = firstVector * secondVector;

	// current
	cout << "Used Value: " << values << " Check Value: " << resultVector[0] << endl;
	auto end = chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed = end - start;

	for (int i = 0; i < count; i++) {
		resultVector[i] = firstVector[i] * secondVector[i];
	}

	auto end2 = chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed2 = end2 - end;


	return "Current: " + to_string(elapsed2.count()) + ", Next?: " + to_string(elapsed.count()) + ", Times Smaller = " + to_string(elapsed2.count() / elapsed.count());
}

string multiplyVectorsTimeTest(int count, float values) {
	auto start = chrono::high_resolution_clock::now();

	valarray<float> firstVector(values, count);
	valarray<float> secondVector(values, count);
	valarray<float> resultVector = firstVector + secondVector;

	// current
	cout << "Used Value: " << values << " Check Value: " << resultVector[0] << endl;
	auto end = chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed = end - start;

	for (int i = 0; i < count; i++) {
		resultVector[i] = firstVector[i] + secondVector[i];
	}

	auto end2 = chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed2 = end2 - end;


	return "Current: " + to_string(elapsed2.count()) + ", Next?: " + to_string(elapsed.count()) + ", Times Smaller = " + to_string(elapsed2.count() / elapsed.count());
}

int main() {

	cout << addVectorsTimeTest(10000000, 10.0f) << endl;
	cout << multiplyVectorsTimeTest(1000000, 10.0f) << endl;

	system("pause");
	return 0;
}