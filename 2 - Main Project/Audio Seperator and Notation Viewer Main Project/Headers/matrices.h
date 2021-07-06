#ifndef MATRICES_H
#define MATRICES_H

/* 
MATRIX FORMAT---------------
-stored as vector<vector<float>>
-each vector<float> is a row
*/

#include <vector>
#include <iostream>
using namespace std;

#define Matrix vector<vector<float>>

Matrix identityMatrix(int size);
Matrix transposeMatrix(Matrix A);

Matrix scalarMultiply(float scalar, Matrix A);
Matrix matrixMultiply(Matrix A, Matrix B); // Order Matters

Matrix matrixAddition(Matrix A, Matrix B);
Matrix inverseMatrix(Matrix A, int decimalTolerance);

#endif // !MATRICES_H
