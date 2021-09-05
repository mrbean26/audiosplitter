#ifndef NAIVE_BAYES_H
#define NAIVE_BAYES_H

#include <vector>
using namespace std;

#define DATASET_TYPE pair<vector<vector<float>>, vector<vector<float>>>
#define DATASET_ENTRY pair<vector<float>, vector<float>>

struct bayesProbability {
	float a = 0.0f;
	float b = 0.0f;

	float bGivenA = 0.0f;
};

bayesProbability getProbabilities(DATASET_TYPE dataset, bool(*a)(DATASET_ENTRY), bool(*b)(DATASET_ENTRY));
float naiveBayes(bayesProbability probabilities);

#endif // !NAIVE_BAYES_H
