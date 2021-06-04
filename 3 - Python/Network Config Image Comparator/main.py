from PIL import Image

import glob
fileNames = glob.glob("*.jpg")

bestNetworkFilename = ""
bestNetworkScore = 0

# CONSTANT TO CHANGE
deltaXIndex = 30

for fileName in fileNames:
    # Load Metadata
    openedFile = open(fileName, 'rb')
    lines = openedFile.readlines()
    openedFile.close()

    finalLine = lines[-1 :][0].decode().strip()
    scaleLow = float(finalLine[finalLine.find(":") + 2 : finalLine.find(",")])

    finalLine = finalLine[finalLine.find(",") + 2 :]
    scaleHigh = float(finalLine[finalLine.find(":") + 2 : finalLine.find(",")])

    # Load Image
    imageFile = Image.open(fileName)
    imageSize = imageFile.size
    pixels = imageFile.load()

    # Find Brightest Pixel in Last Column (Lowest Error)
    brighestPixelIndex = 0
    brighestPixelValue = 0

    for i in range(imageSize[1]):
        currentValue = pixels[imageSize[0] - 1, i]

        if currentValue[0] > brighestPixelValue:
            brighestPixelValue = currentValue[0]
            brighestPixelIndex = i

    brighestPixelIndex = (imageSize[1] - 1) - brighestPixelIndex
    finalError = (float(brighestPixelIndex) / float(imageSize[1])) * scaleHigh + scaleLow

    # Find Brightest Pixel in Delta Column
    brighestPixelIndex = 0
    brighestPixelValue = 0

    for i in range(imageSize[1]):
        currentValue = pixels[imageSize[0] - 1 - deltaXIndex, i]

        if currentValue[0] > brighestPixelValue:
            brighestPixelValue = currentValue[0]
            brighestPixelIndex = i

    brighestPixelIndex = (imageSize[1] - 1) - brighestPixelIndex
    deltaError = (float(brighestPixelIndex) / float(imageSize[1])) * scaleHigh + scaleLow

    # Calculate Score
    if finalError == 0:
        bestNetworkFilename = fileName
        break

    currentScore = (1 / finalError) * ((finalError - deltaError) / float(deltaXIndex))

    if currentScore > bestNetworkScore:
        bestNetworkScore = currentScore
        bestNetworkFilename = fileName

print("Best Network:", bestNetworkFilename)
