#include "Headers/KNearestNeighbors.h"

float getElucidianDistance(vector<float> one, vector<float> two) {
	int size = one.size();
	if (size != two.size()) {
		return -1.0f;
	}

	float accumulativeTotal = 0.0f;
	for (int i = 0; i < size; i++) {
		float difference = one[i] - two[i];
		difference = powf(difference, 2.0f);

		accumulativeTotal = accumulativeTotal + difference;
	}

	return sqrtf(accumulativeTotal);
}
vector<float> addVector(vector<float> one, vector<float> two) {
	int size = one.size();
	if (size != two.size()) {
		return {};
	}

	for (int i = 0; i < size; i++) {
		one[i] = one[i] + two[i];
	}

	return one;
}
vector<float> multiplyVector(float scalar, vector<float> one) {
	int size = one.size();

	for (int i = 0; i < size; i++) {
		one[i] = one[i] * scalar;
	}

	return one;
}
bool checkVectorEquality(vector<float> one, vector<float> two) {
	int size = one.size();
	if (size != two.size()) {
		return false;
	}

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if (one[i] != two[i]) {
				return false;
			}
		}
	}

	return true;
}
vector<float> mostAppearancesInVector(vector<vector<float>> one) {
	int size = one.size();

	vector<float> mostCommon;
	int mostCommonCount = 0;

	for (int i = 0; i < size; i++) {
		vector<float> current = one[i];
		int currentCount = 1;

		for (int j = 0; j < size; j++) {
			if (j == i) {
				continue;
			}

			if (checkVectorEquality(current, one[j])) {
				currentCount = currentCount + 1;
			}
		}

		if (currentCount > mostCommonCount) {
			mostCommonCount = currentCount;
			mostCommon = current;
		}
	}

	return mostCommon;
}

vector<float> kNearestNeighbors(pair<vector<vector<float>>, vector<vector<float>>> dataset, int kParameter, int outputType, vector<float> testPoint) {
	pair<vector<float>, vector<vector<float>>> nearestPoints; // First is Distance From TestPoint and Second is Point
	int dataSetSize = dataset.first.size();

	for (int i = 0; i < dataSetSize; i++) {
		float currentDistance = getElucidianDistance(dataset.first[i], testPoint);
		
		// Add Point To Vector if Distance is Low Enough
		int currentSize = nearestPoints.first.size();

		if (currentSize == kParameter) {
			if (currentDistance < nearestPoints.first[currentSize - 1]) {
				nearestPoints.first[kParameter - 1] = currentDistance;
				nearestPoints.second[kParameter - 1] = dataset.second[i];
			}
		}
		else {
			nearestPoints.first.push_back(currentDistance);
			nearestPoints.second.push_back(dataset.second[i]);
		}

		// Sort by Distance
		for (int i = currentSize - 2; i >= 0; i--) {
			if (currentDistance < nearestPoints.first[i]) {
				float tmpDistance = nearestPoints.first[i];
				vector<float> tmpPoint = nearestPoints.second[i];

				nearestPoints.first[i] = currentDistance;
				nearestPoints.second[i] = dataset.second[i];

				nearestPoints.first[i + 1] = tmpDistance;
				nearestPoints.second[i + 1] = tmpPoint;
			}
		}
	}
	
	// Return Prediction or Classification
	if (outputType == CLASSIFICATION_OUTPUT) {
		vector<vector<float>> allClasses;
		for (int i = 0; i < kParameter; i++) {
			allClasses.push_back(nearestPoints.second[i]);
		}

		return mostAppearancesInVector(allClasses);
	}
	if (outputType == REGRESSION_OUTPUT) {
		if (kParameter == 1) {
			return nearestPoints.second[0];
		}

		float totalDistance = 0.0f;
		for (int i = 0; i < kParameter; i++) {
			totalDistance = totalDistance + nearestPoints.first[i];
		}
		
		// Take Mean of Nearest Points According To Distance (like isotope abundance formula)
		vector<float> result(nearestPoints.second[0].size());
		for (int i = 0; i < kParameter; i++) {
			vector<float> currentResult = multiplyVector(totalDistance - nearestPoints.first[i], nearestPoints.second[i]);
			result = addVector(result, currentResult);
		}

		result = multiplyVector(1.0f / totalDistance, result);
		return result;
	}
}