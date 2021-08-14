# Base Libraries
from random import uniform

bias = uniform(-10.0, 10.0)
vector = [uniform(-10.0, 10.0), uniform(-10.0, 10.0), uniform(-10.0, 10.0), uniform(-10.0, 10.0), uniform(-10.0, 10.0)]

# Load Part of IRIS Dataset
inputs = []
outputs = []

irisFile = open("Iris.csv")
irisLines = irisFile.readlines()

for i in range(100):
    index = i
    data = irisLines[index].split(",")

    inputs.append([float(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4])])

    if(data[5] == "Iris-setosa\n"):
        outputs.append(1)
    else:
        outputs.append(-1)

# Vector Math
def scalarProduct(a, b):
    result = [0] * len(a)


    for i in range(len(a)):
        result[i] = a[i] * b

    return result

def dotProduct(a, b):
    result = 0

    for i in range(len(a)):
        for j in range(len(b)):
            result += a[i] * b[i]

    return result

def subtractVector(a, b):
    result = []

    for i in range(len(b)):
        result.append(a[i] - b[i])

    return result

# Find Support Vector
epochs = 1000
LR = 0.001

for i in range(epochs):
    for j in range(len(inputs)):
        if 1 - outputs[j] * (dotProduct(vector, inputs[j]) + bias) > 0:
            bias = bias - LR * (-outputs[j])

        main = scalarProduct(vector, 2)

        if 1 - outputs[j] * (dotProduct(vector, inputs[j]) + bias) > 0:
            value = 1 / (len(inputs)) * outputs[j]

            scaled = scalarProduct(inputs[j], value)
            main = subtractVector(main, scaled)

        main = scalarProduct(main, LR)
        vector = subtractVector(vector, main)

# Run Tests
counts = [0, 0]

for i in range(len(inputs)):
    product = dotProduct(inputs[i], vector)
    product += bias

    if round(product) == 1:
        counts[0] += 1
    else:
        counts[1] += 1

print(counts)
