# Matrix Math
def zeros_matrix(rows, cols):
    A = []
    for i in range(rows):
        A.append([])
        for j in range(cols):
            A[-1].append(0.0)

    return A

def copy_matrix(M):
    rows = len(M)
    cols = len(M[0])

    MC = zeros_matrix(rows, cols)

    for i in range(rows):
        for j in range(rows):
            MC[i][j] = M[i][j]

    return MC

def transpose(M):
    """
    Returns a transpose of a matrix.
        :param M: The matrix to be transposed

        :return: The transpose of the given matrix
    """
    # Section 1: if a 1D array, convert to a 2D array = matrix
    if not isinstance(M[0],list):
        M = [M]

    # Section 2: Get dimensions
    rows = len(M)
    cols = len(M[0])

    # Section 3: MT is zeros matrix with transposed dimensions
    MT = zeros_matrix(cols, rows)

    # Section 4: Copy values from M to it's transpose MT
    for i in range(rows):
        for j in range(cols):
            MT[j][i] = M[i][j]

    return MT

def matrix_multiply(A, B):
    """
    Returns the product of the matrix A * B
        :param A: The first matrix - ORDER MATTERS!
        :param B: The second matrix

        :return: The product of the two matrices
    """
    # Section 1: Ensure A & B dimensions are correct for multiplication
    rowsA = len(A)
    colsA = len(A[0])
    rowsB = len(B)
    colsB = len(B[0])
    if colsA != rowsB:
        raise ArithmeticError(
            'Number of A columns must equal number of B rows.')

    # Section 2: Store matrix multiplication in a new matrix
    C = zeros_matrix(rowsA, colsB)
    for i in range(rowsA):
        for j in range(colsB):
            total = 0
            for ii in range(colsA):
                total += A[i][ii] * B[ii][j]
            C[i][j] = total

    return C

def scalar_multiply(k, M):
    result = M

    rows = len(M)
    cols = len(M[0])

    for i in range(rows):
        for j in range(cols):
            result[i][j] = M[i][j] * k

    return result

def identity_matrix(n):
    """
    Creates and returns an identity matrix.
        :param n: the square size of the matrix

        :return: a square identity matrix
    """
    IdM = zeros_matrix(n, n)
    for i in range(n):
        IdM[i][i] = 1.0

    return IdM

def check_matrix_equality(A, B, tol=None):
    """
    Checks the equality of two matrices.
        :param A: The first matrix
        :param B: The second matrix
        :param tol: The decimal place tolerance of the check

        :return: The boolean result of the equality check
    """
    # Section 1: First ensure matrices have same dimensions
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        return False

    # Section 2: Check element by element equality
    #            use tolerance if given
    for i in range(len(A)):
        for j in range(len(A[0])):
            if tol == None:
                if A[i][j] != B[i][j]:
                    return False
            else:
                if round(A[i][j],tol) != round(B[i][j],tol):
                    return False

    return True

def invert_matrix(A, tol=None):
    """
    Returns the inverse of the passed in matrix.
        :param A: The matrix to be inversed

        :return: The inverse of the matrix A
    """
    # Section 1: Make sure A can be inverted.
    #check_squareness(A)
    #check_non_singular(A)

    # Section 2: Make copies of A & I, AM & IM, to use for row ops
    n = len(A)
    AM = copy_matrix(A)
    I = identity_matrix(n)
    IM = copy_matrix(I)

    # Section 3: Perform row operations
    indices = list(range(n)) # to allow flexible row referencing ***
    for fd in range(n): # fd stands for focus diagonal
        fdScaler = 1.0 / AM[fd][fd]
        # FIRST: scale fd row with fd inverse.
        for j in range(n): # Use j to indicate column looping.
            AM[fd][j] *= fdScaler
            IM[fd][j] *= fdScaler
        # SECOND: operate on all rows except fd row as follows:
        for i in indices[0:fd] + indices[fd+1:]:
            # *** skip row with fd in it.
            crScaler = AM[i][fd] # cr stands for "current row".
            for j in range(n):
                # cr - crScaler * fdRow, but one element at a time.
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
                IM[i][j] = IM[i][j] - crScaler * IM[fd][j]

    # Section 4: Make sure IM is an inverse of A with specified tolerance
    if check_matrix_equality(I,matrix_multiply(A,IM),tol):
        return IM
    else:
        raise ArithmeticError("Matrix inverse out of tolerance.")

def matrix_addition(A, B):
    """
    Adds two matrices and returns the sum
        :param A: The first matrix
        :param B: The second matrix

        :return: Matrix sum
    """
    # Section 1: Ensure dimensions are valid for matrix addition
    rowsA = len(A)
    colsA = len(A[0])
    rowsB = len(B)
    colsB = len(B[0])
    if rowsA != rowsB or colsA != colsB:
        raise ArithmeticError('Matrices are NOT the same size.')

    # Section 2: Create a new matrix for the matrix sum
    C = zeros_matrix(rowsA, colsB)

    # Section 3: Perform element by element sum
    for i in range(rowsA):
        for j in range(colsB):
            C[i][j] = A[i][j] + B[i][j]

    return C

# Example Problem Functions
import math

def getError(actual, a, b, x):
    functionOutput = a * math.cos(b * x) + b * math.sin(a * x)

    return desiredOutput - functionOutput

def getJacobianMatrix(a, b, x):
    result = [[]]

    # Add Derivatives (WITH RESPECT TO "ACTUAL - PREDICTED")
    result[0].append(-b * x * math.cos(a * x) - math.cos(b * x)) # A
    result[0].append(a * x * math.sin(b * x) - math.sin(a * x)) # B
    result[0].append(a * b * math.sin(b * x) - a * b * math.cos(a * x)) # X

    return result

# Algorithm
dampValue = 0.001

import random
A = random.randint(1, 100)
B = random.randint(1, 100)
X = random.randint(1, 100)

iterations = 10000
lowestDamp = False

desiredOutput = 42
finalError = 0

for i in range(iterations):
    previousError = getError(desiredOutput, A, B, X)

    jacobian = getJacobianMatrix(A, B, X)
    transposedJacobian = transpose(jacobian)

    approximateHessian = matrix_multiply(transposedJacobian, jacobian)

    identityMatrix = identity_matrix(len(jacobian[0]))
    identityMatrix = scalar_multiply(dampValue, identityMatrix)

    approximateHessian = matrix_addition(approximateHessian, identityMatrix)
    try:
        approximateHessian = invert_matrix(approximateHessian, 0)
    except:
        # Inverse Not Calculable (Matrix is singular due to v low damp)
        dampValue = dampValue * 10
        lowestDamp = True
        continue

    approximateHessian = scalar_multiply(-1, approximateHessian)

    deltas = matrix_multiply(approximateHessian, transposedJacobian)
    deltas = scalar_multiply(previousError, deltas)

    A = A + deltas[0][0]
    B = B + deltas[1][0]
    X = X + deltas[2][0]

    if getError(desiredOutput, A, B, X) > previousError:
        dampValue = dampValue * 10
        lowestDamp = False

        A = A - deltas[0][0]
        B = B - deltas[1][0]
        X = X - deltas[2][0]
    else:
        if lowestDamp == False:
            dampValue = dampValue * 0.1

    finalError = previousError

print("Final Error:", finalError)
print("Parameters:", A, B, X)
