from math import sqrt, cos, sin

lr = 0.001
epsillon = 0.0000001
betaParameter = 0.9
epochs = 500
batchSize = 200

a = 0.7
b = 0.2

def getError():
    return pow(a * cos(b) + b * sin(a), 2)

def getDerivatives():
    return [
        2 * (a * cos(b) + b * cos(a)) * (cos(b) + b * cos(a)),
        2 * (a * cos(b) + b * sin(a)) * (-a * sin(b) + sin(a))
    ]


for epoch in range(epochs):
    meanSquare = [0, 0]
    meanGradients = [0, 0]

    for i in range(batchSize):
        currentError = getError()
        gradients = getDerivatives()

        meanSquare[0] = betaParameter * meanSquare[0] + (1 - betaParameter) * pow(gradients[0], 2)
        meanSquare[1] = betaParameter * meanSquare[1] + (1 - betaParameter) * pow(gradients[1], 2)

        meanGradients[0] += gradients[0] / batchSize
        meanGradients[1] += gradients[1] / batchSize

    deltas = [
        meanGradients[0] / (sqrt(meanSquare[0]) + epsillon),
        meanGradients[1] / (sqrt(meanSquare[1]) + epsillon)
    ]

    a = a - lr * deltas[0]
    b = b - lr * deltas[1]

print("Final Error:", currentError, "A:", a, "B:", b)
