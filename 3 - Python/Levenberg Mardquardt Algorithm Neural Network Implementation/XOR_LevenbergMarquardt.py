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

def identity_matrix(n, k = 1):
    """
    Creates and returns an identity matrix.
        :param n: the square size of the matrix

        :return: a square identity matrix
    """
    IdM = zeros_matrix(n, n)
    for i in range(n):
        IdM[i][i] = 1.0 * k

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
    for fd in range(n): # fd stands for focus diagonal
        fdScaler = 1.0 / AM[fd][fd]
        # FIRST: scale fd row with fd inverse.
        for j in range(n): # Use j to indicate column looping.
            AM[fd][j] *= fdScaler
            IM[fd][j] *= fdScaler

        # SECOND: operate on all rows except fd row as follows:

        for i in range(n):
            if i == fd:
                continue
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

# Standard Math Functions
import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def deriveSigmoid(x):
    return x * (1 - x)

def getWeight():
    return (random.random() * 2 - 1)

# Network Nodes
inputOne = 0
inputTwo = 0

hiddenOne = 0
hiddenTwo = 0

outputOne = 0

biasOne = 0
hiddenBias = 0

# Network Derivatives
hiddenOneDerivative = 0
hiddenTwoDerivative = 0
outputDerivative = 0

# Weights
inputWeights = [getWeight(), getWeight(), getWeight(), getWeight(), getWeight(), getWeight()]
hiddenWeights = [getWeight(), getWeight(), getWeight()]

# Network Functions
def feedForward(a, b):
    global inputOne
    global inputTwo
    global hiddenOne
    global hiddenTwo
    global outputOne

    inputOne = a
    inputTwo = b

    hiddenOne = inputOne * inputWeights[0] + inputTwo * inputWeights[2] + inputWeights[4]
    hiddenTwo = inputOne * inputWeights[1] + inputTwo * inputWeights[3] + inputWeights[5]

    hiddenOne = sigmoid(hiddenOne)
    hiddenTwo = sigmoid(hiddenTwo)

    outputOne = hiddenOne * hiddenWeights[0] + hiddenTwo * hiddenWeights[1] + hiddenWeights[2]
    outputOne = sigmoid(outputOne)

def findDerivatives(error):
    global outputDerivative
    global hiddenOneDerivative
    global hiddenTwoDerivative

    outputDerivative = deriveSigmoid(outputOne) * error * -2

    hiddenOneDerivative = deriveSigmoid(hiddenOne) * hiddenWeights[0] * outputDerivative
    hiddenTwoDerivative = deriveSigmoid(hiddenTwo) * hiddenWeights[1] * outputDerivative

def getJacobianMatrixRow():
    # Order: Input weights and then hidden
    jacobianMatrix = []

    jacobianMatrix.append(inputOne * hiddenOneDerivative)
    jacobianMatrix.append(inputOne * hiddenTwoDerivative)

    jacobianMatrix.append(inputTwo * hiddenOneDerivative)
    jacobianMatrix.append(inputTwo * hiddenTwoDerivative)

    jacobianMatrix.append(hiddenOneDerivative)
    jacobianMatrix.append(hiddenTwoDerivative)

    jacobianMatrix.append(hiddenOne * outputDerivative)
    jacobianMatrix.append(hiddenTwo * outputDerivative)
    jacobianMatrix.append(outputDerivative)

    return jacobianMatrix

def addDeltas(matrixDeltas):
    global inputWeights
    global hiddenWeights

    inputWeights[0] -= matrixDeltas[0][0]
    inputWeights[1] -= matrixDeltas[1][0]
    inputWeights[2] -= matrixDeltas[2][0]
    inputWeights[3] -= matrixDeltas[3][0]
    inputWeights[4] -= matrixDeltas[4][0]
    inputWeights[5] -= matrixDeltas[5][0]

    hiddenWeights[0] -= matrixDeltas[6][0]
    hiddenWeights[1] -= matrixDeltas[7][0]
    hiddenWeights[2] -= matrixDeltas[8][0]

# Train (Levenberg Marquardt)
dampValue = 0.01
dampIncreaseMultiplier = 10
dampDecreaseMultiplier = 0.1

epochs = 10000

trainInputs = [[0, 0], [1, 0], [0, 1], [1, 1]]
trainOutputs = [0, 1, 1, 0]

lowestDamp = False
for epoch in range(epochs):
    currentTotalJacobian = []
    currentCosts = []
    currentTotalCost = 0

    for t in range(4):
        feedForward(trainInputs[t][0], trainInputs[t][1])

        currentError = trainOutputs[t] - outputOne
        currentCost = currentError ** 2
        currentTotalCost += currentCost
        findDerivatives(currentError)

        currentJacobianRow = getJacobianMatrixRow()
        currentTotalJacobian.append(currentJacobianRow)
        currentCosts.append([currentCost])

    # Update
    transposedJacobian = transpose(currentTotalJacobian)
    identityMatrix = identity_matrix(len(currentTotalJacobian[0]), dampValue)

    approximateHessian = matrix_addition(matrix_multiply(transposedJacobian, currentTotalJacobian), identityMatrix)
    try:
        approximateHessian = invert_matrix(approximateHessian, 0)
    except:
        dampValue = dampValue * dampIncreaseMultiplier
        continue

    transposedJacobian = matrix_multiply(transposedJacobian, currentCosts)
    transposedJacobian = scalar_multiply(2, transposedJacobian)

    currentDeltas = matrix_multiply(approximateHessian, transposedJacobian)
    addDeltas(currentDeltas)

    # Check if error is lower
    newTotalCost = 0
    for t in range(4):
        feedForward(trainInputs[t][0], trainInputs[t][1])

        currentError = trainOutputs[t] - outputOne
        currentCost = currentError ** 2
        newTotalCost += currentCost

    if newTotalCost > currentTotalCost:
        dampValue = dampValue * dampIncreaseMultiplier

        currentDeltas = scalar_multiply(-1, currentDeltas)
        addDeltas(currentDeltas)
    else:
        dampValue = dampValue * dampDecreaseMultiplier

for i in range(4):
    feedForward(trainInputs[i][0], trainInputs[i][1])
    print(outputOne)
