#include "Headers/matrices.h"

// General
float roundFloat(float value, int places) {
	float roundInput = value * (10 * places);
	float rounded = roundf(roundInput);

	return rounded / (10 * places);
}
bool checkMatrixEquality(Matrix A, Matrix B, int decimalTolerance) {
	int rowsA = A.size();
	int columnsA = A[0].size();

	int rowsB = B.size();
	int columnsB = B[0].size();

	if (decimalTolerance == 0) {
		for (int i = 0; i < rowsA; i++) {
			for (int j = 0; j < columnsA; j++) {
				if (roundf(A[i][j]) != roundf(B[i][j])) {
					return false;
				}
			}
		}

		return true;
	}

	if (rowsA != rowsB || columnsA != columnsB) {
		return false;
	}

	for (int i = 0; i < rowsA; i++) {
		for (int j = 0; j < columnsA; j++) {
			if (roundFloat(A[i][j], decimalTolerance) != roundFloat(B[i][j], decimalTolerance)) {
				return false;
			}
		}
	}
	return true;
}
Matrix zerosMatrix(int rows, int columns) {
	Matrix returned;

	for (int i = 0; i < rows; i++) {
		vector<float> newRow(columns);
		returned.push_back(newRow);
	}

	return returned;
}

// Transpose & Identity
Matrix identityMatrix(int size) {
	Matrix result = zerosMatrix(size, size);

	for (int i = 0; i < size; i++) {
		result[i][i] = 1.0f;
	}

	return result;
}
Matrix transposeMatrix(Matrix A) {
	int rows = A.size();
	int columns = A[0].size();

	Matrix result = zerosMatrix(columns, rows);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			result[j][i] = A[i][j];
		}
	}

	return result;
}

// Multiplication
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
Matrix matrixMultiply(Matrix A, Matrix B) {
	int rowsA = A.size();
	int columnsA = A[0].size();

	int rowsB = B.size();
	int columnsB = B[0].size();

	if (columnsA != rowsB) {
		cout << "Columns A does not equal Rows B" << endl;
		return Matrix();
	}

	Matrix result = zerosMatrix(rowsA, columnsB);
	for (int i = 0; i < rowsA; i++) {
		for (int j = 0; j < columnsB; j++) {
			float total = 0.0f;

			for (int k = 0; k < columnsA; k++) {
				total = total + A[i][k] * B[k][j];
			}

			result[i][j] = total;
		}
	}

	return result;
}

// Addition & Inverse
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
Matrix inverseMatrix(Matrix A, int decimalTolerance) {
	int n = A.size();

	Matrix AM = A;
	Matrix I = identityMatrix(n);
	Matrix IM = I;

	for (int fd = 0; fd < n; fd++) {
		float fdScalar = 1.0f / AM[fd][fd];
		for (int j = 0; j < n; j++) {
			AM[fd][j] *= fdScalar;
			IM[fd][j] *= fdScalar;
		}

		for (int i = 0; i < n; i++) {
			if (i == fd) {
				continue;
			}

			float currentRowScalar = AM[i][fd];
			for (int j = 0; j < n; j++) {
				AM[i][j] = AM[i][j] - currentRowScalar * AM[fd][j];
				IM[i][j] = IM[i][j] - currentRowScalar * IM[fd][j];
			}
		}
	}
	
	Matrix remultipliedCheck = matrixMultiply(A, IM);
	if (checkMatrixEquality(I, remultipliedCheck, decimalTolerance)) {
		return IM;
	}
	else {
		cout << "Matrix inverse out of tolerance" << endl;
		return {  };
	}
}
