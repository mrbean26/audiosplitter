#ifndef FILESHADERS_H
#define FILESHADERS_H

#include <glm.hpp>
#include <glm/gtc/type_ptr.hpp>
using namespace glm;

#include <iostream>
#include <fstream>
#include <string>

#include <vector>
using namespace std;

#include <glew.h>
#include <glfw3.h>

vector<string> readLines(const char* fileName);
void setMat4(int shader, const char* matrixName, mat4 usedMatrix);

int createShader(const char* filePath, GLenum shaderType);
int createProgram(vector<int> shaders);

void setShaderInt(int shader, const char* intName, int usedInt);

#endif // !FILESHADERS_H

