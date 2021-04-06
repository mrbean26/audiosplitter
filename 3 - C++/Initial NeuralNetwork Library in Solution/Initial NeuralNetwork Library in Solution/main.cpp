#include "Headers/NeuralNetwork.h"
#include <time.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void writeToImage(vector<float> errors, int errorResolution, int errorRange) {
    // Normalise Errors
    float maxError = 0.0f;
    int errorCount = errors.size();

    for (int i = 0; i < errorCount; i++) {
        maxError = max(maxError, errors[i]);
    }

    for (int i = 0; i < errorCount; i++) {
        errors[i] = (errors[i] / maxError) * (errorRange - 1);
    }

    // Average Out Error Pixels
    int vectorsPerPixel = errorCount / errorResolution;
    vector<vector<float>> pixelValues;

    for (int i = 0; i < errorCount; i += vectorsPerPixel) {
        vector<float> current(errorRange);

        for (int j = 0; j < vectorsPerPixel; j++) {
            current[int(errors[i + j])] += 1;
        }

        for (int j = 0; j < vectorsPerPixel; j++) {
            current[j] = current[j] / vectorsPerPixel;
        }

        pixelValues.push_back(current);
    }

    // Write To Image
    unsigned char * data = new unsigned char[errorResolution * errorRange * 3];
    int index = 0;

    for (int y = errorRange - 1; y >= 0; y -= 1) {
        for (int x = 0; x < errorResolution; x++) {
            data[index++] = (unsigned char)(255.0 * pixelValues[x][y]);
            data[index++] = (unsigned char)(255.0 * pixelValues[x][y]);
            data[index++] = (unsigned char)(255.0 * pixelValues[x][y]);
        }
    }

    stbi_write_jpg("errorOutput.jpg", errorResolution, errorRange, 3, data, errorResolution * 3);
}

int main(){
    srand(time(NULL));
    NeuralNetwork net = NeuralNetwork({2, 2, 1}, {1, 1, 0}, "tanh");

    vector<vector<float>> inputs = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};
    vector<vector<float>> outputs = {{0.0f}, {1.0f}, {1.0f}, {0.0f}};

    net.loadWeightsFromFile("outputWeights/");

    vector<float> returnedErrors = net.train(inputs, outputs, 10000, 0.1f, 0.5f);
    writeToImage(returnedErrors, 1000, 512);

    net.runTests(inputs);
    net.saveWeightsToFile("outputWeights/");

    system("pause");
    return 0;
}
