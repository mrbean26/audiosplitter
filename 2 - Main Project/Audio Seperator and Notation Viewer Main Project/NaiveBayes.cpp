#include "Headers/NaiveBayes.h"

bayesProbability getProbabilities(DATASET_TYPE dataset, bool(*a)(DATASET_ENTRY), bool(*b)(DATASET_ENTRY)) {
	bayesProbability result = bayesProbability();

	// Accumulate Occurrences
	int totalDatasetSize = dataset.first.size();

	for (int i = 0; i < totalDatasetSize; i++) {
		DATASET_ENTRY currentEntry = make_pair(dataset.first[i], dataset.second[i]);

		if (a(currentEntry)) {
			result.a = result.a + 1.0f;

			if (b(currentEntry)) {
				result.bGivenA = result.bGivenA + 1.0f;
				result.b = result.b + 1.0f;
				continue;
			}
		}
		if (b(currentEntry)) {
			result.b = result.b + 1.0f;
		}
	}

	// Calculate Probabilities
	if (result.a != 0.0f) {
		result.bGivenA = result.bGivenA / result.a;
	}
	result.a = result.a / float(totalDatasetSize);
	result.b = result.b / float(totalDatasetSize);

	return result;
}

float naiveBayes(bayesProbability probabilities) {
	return (probabilities.bGivenA * probabilities.a) / probabilities.b;
}