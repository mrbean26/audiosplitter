#ifndef K_NEAREST_NEIGHBORS
#define K_NEAREST_NEIGHBORS

#include <vector>
#include <iostream>
using namespace std;

#define CLASSIFICATION_OUTPUT 0
#define REGRESSION_OUTPUT 1

vector<float> kNearestNeighbors(pair<vector<vector<float>>, vector<vector<float>>> dataset, int kParameter, int outputType, vector<float> testPoint);

#endif // !K_NEAREST_NEIGHBORS
