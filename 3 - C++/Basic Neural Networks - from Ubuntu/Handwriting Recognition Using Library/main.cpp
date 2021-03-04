#include "Headers/NeuralNetwork.h"

#include <experimental/filesystem> // Make sure you add -lstdc++fs
namespace fs = experimental::filesystem;

#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "Headers/stb_image.h"

vector<float> blackWhiteImageToInputs(const char * fileName){ // 1.0f for white, 0.0f for black
    int x, y, n;
    unsigned char * data = stbi_load(fileName, &x, &y, &n, 0);

    vector<int> currentPixelValues = {0, 0, 0, 0};
    vector<float> result;

    for(int i = 0; i < x * y * n; i++){
        currentPixelValues[i % n] = static_cast<int>(data[i]);

        if(i % n == 0 && i > 0){
            if(currentPixelValues[0] > 122){
                result.push_back(1.0f);
                continue;
            }
            result.push_back(0.0f);
        }
    }
    stbi_image_free(data);

    return result;
}

vector<vector<vector<float>>> dataFromDirectory(string directory, int outputIndex){
    vector<vector<float>> resultInputs;
    vector<vector<float>> resultOutputs;

    for(const auto & entry : fs::directory_iterator(directory)){
        const char * newPath = entry.path().c_str();

        resultInputs.push_back(blackWhiteImageToInputs(newPath));

        vector<float> newResults(36);
        newResults[outputIndex] = 1.0f;
        resultOutputs.push_back(newResults);
    }

    return {resultInputs, resultOutputs};
}

vector<vector<vector<float>>> getData(){
    vector<string> allFolderNames = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P" ,"Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};

    vector<vector<float>> resultInputs;
    vector<vector<float>> resultOutputs;

    int count = allFolderNames.size();
    for(int i = 0; i < count; i++){
        vector<vector<vector<float>>> currentResult = dataFromDirectory("Dataset/" + allFolderNames[i] + "/", i);

        resultInputs.insert(resultInputs.end(), currentResult[0].begin(), currentResult[0].end());
        resultOutputs.insert(resultOutputs.end(), currentResult[1].begin(), currentResult[1].end());
    }

    return {resultInputs, resultOutputs};
}

string getCharacterFromProbablilities(vector<float> prediction){
    vector<string> characters = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};

    float currentHighest = 0.0f;
    int currentHighestIndex = 0;

    int size = prediction.size();
    for(int i = 0; i < size; i++){
        if(prediction[i] > currentHighest){
            currentHighest = prediction[i];
            currentHighestIndex = i;
        }
    }

    return characters[currentHighestIndex];
}

int main(){
    // Run with "g++ main.cpp NeuralNetwork.cpp -lstdc++fs -O3"
    srand(time(NULL));

    vector<vector<vector<float>>> trainingData = getData();
    vector<vector<float>> trainInputs = trainingData[0];
    vector<vector<float>> trainOutputs = trainingData[1];

    // Network, could use some more training as it gets confused with numbers
    int inputNodeCount = trainInputs[0].size();
    NeuralNetwork network = NeuralNetwork({inputNodeCount, inputNodeCount * 2, 36}, {1, 1, 0}, "sigmoid");
    //network.loadWeightsFromFile("");
    network.train(trainInputs, trainOutputs, 5000, 0.1f, 0.5f);
    network.saveWeightsToFile("");
    //network.runTests(trainInputs);
}
