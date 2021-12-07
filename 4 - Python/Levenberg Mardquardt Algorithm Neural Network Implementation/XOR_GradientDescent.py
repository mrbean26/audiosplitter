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

    outputDerivative = deriveSigmoid(outputOne) * error

    hiddenOneDerivative = deriveSigmoid(hiddenOne) * hiddenWeights[0] * outputDerivative
    hiddenTwoDerivative = deriveSigmoid(hiddenTwo) * hiddenWeights[1] * outputDerivative

def updateWeightsGradientDescent(learningRate):
    global inputWeights
    global hiddenWeights

    # Input Layer
    inputWeights[0] += learningRate * inputOne * hiddenOneDerivative
    inputWeights[1] += learningRate * inputOne * hiddenTwoDerivative

    inputWeights[2] += learningRate * inputTwo * hiddenOneDerivative
    inputWeights[3] += learningRate * inputTwo * hiddenTwoDerivative

    inputWeights[4] += learningRate * hiddenOneDerivative
    inputWeights[5] += learningRate * hiddenTwoDerivative

    # Hidden Layer
    hiddenWeights[0] += learningRate * hiddenOne * outputDerivative
    hiddenWeights[1] += learningRate * hiddenTwo * outputDerivative
    hiddenWeights[2] += learningRate * outputDerivative

# Train Here (Gradient Descent)
learningRate = 0.1
epochs = 40000

trainInputs = [[0, 0], [1, 0], [0, 1], [1, 1]]
trainOutputs = [0, 1, 1, 0]

for epoch in range(epochs):
    feedForward(trainInputs[epoch % 4][0], trainInputs[epoch % 4][1])

    error = trainOutputs[epoch % 4] - outputOne
    findDerivatives(error)

    updateWeightsGradientDescent(learningRate)

for i in range(4):
    feedForward(trainInputs[i][0], trainInputs[i][1])
    print(outputOne)
