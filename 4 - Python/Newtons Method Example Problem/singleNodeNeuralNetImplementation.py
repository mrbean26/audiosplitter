actual = 1
input = 0.5
weight = 0.6

from math import exp
def activate(x):
    return 1 / (1+exp(-x))

# DERIVATIVES FROM SYMBOLAB
def deriveFirst():
    numerator = exp(-input * weight) * input
    denominator = pow(1 + exp(-input * weight), 2)

    return numerator / denominator

# DERIVATIVES FROM SYMBOLAB
def deriveSecond():
    numerator = exp(-2 * input * weight) * pow(input, 2) * (-exp(input * weight) + 1)
    denominator = pow(1 + exp(-input * weight), 3)

    return numerator / denominator

def findOutput():
    return activate(input * weight)

for i in range(100):
    output = findOutput()
    error = actual - output

    delta = deriveFirst() / deriveSecond()
    weight -= delta

    print(output, error)

print(weight)
