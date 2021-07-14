from random import uniform
from math import cos, sin

x = 2
y = 2
z = 3

def getCost():
    return x ** 2 + 2 * y + z ** 3

epochs = 10
for i in range(epochs):
    currentCost = getCost()
    print(currentCost)
    if 2 * x != 0:
        x = x - (currentCost / (2 * x))

    y = y - (currentCost / (2))

    if pow(3 * z, 2) != 0:
        z = z - (currentCost / (pow(3 * z, 2)))

print("X:", x)
print("Y:", y)
print("Z:", z)

print("Final Cost:", currentCost)
