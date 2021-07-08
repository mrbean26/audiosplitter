from random import uniform
from math import cos, sin

x = 6
y = 2
z = -1200

def getCost():
    return x ** 4 + y ** 3 - 1 + 3 * z

epochs = 1000
for i in range(epochs):
    currentCost = getCost()

    x = x - (currentCost / (pow(4 * x, 3)))
    y = y - (currentCost / (pow(3 * y, 2)))
    z = z - (currentCost / (3))

    if i % 100 == 0:
        print(currentCost)

print("X:", x)
print("Y:", y)
print("Z:", z)
