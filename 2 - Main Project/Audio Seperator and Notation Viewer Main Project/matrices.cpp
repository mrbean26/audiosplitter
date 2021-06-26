#include "Headers/matrices.h"

Matrix zerosMatrix(int rows, int columns) {
	Matrix returned;

	for (int i = 0; i < rows; i++) {
		vector<float> newRow(columns);
		returned.push_back(newRow);
	}

	return returned;
}

Matrix identityMatrix(int size) {
	Matrix result = zerosMatrix(size, size);

	for (int i = 0; i < size; i++) {
		result[i][i] = 1.0f;
	}

	return result;
}

Matrix scalarMultiply(float scalar, Matrix A) {
	Matrix result = A;

	int rows = A.size();
	int columns = A[0].size();

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			result[i][j] = result[i][j] * scalar;
		}
	}

	return result;
}

Matrix transposeMatrix(Matrix A) {
	Matrix result = A;

	int rows = A.size();
	int columns = A[0].size();

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			result[i][j] = A[j][i];
		}
	}

	return result;
}

Matrix matrixAddition(Matrix A, Matrix B) {
	Matrix result = A;

	int rows = A.size();
	int columns = A[0].size();

	int rowsB = B.size();
	int columnsB = B[0].size();

	if (rows != rowsB || columns != columnsB) {
		cout << "Not The Same Size Matrices in Addition" << endl;
	}

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			result[i][j] = result[i][j] + B[i][j];
		}
	}

	return result;
}
