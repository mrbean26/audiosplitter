#include "Headers/texture.h"

#include <glew.h>
#include <glfw3.h>

#include "stb_image.h" //image library

#include <iostream>
using namespace std;

texture loadTexture(const char * filePath){
	texture newTexture;
	glGenTextures(1, &newTexture.textureId);
	newTexture.data = stbi_load(filePath, &newTexture.width, &newTexture.height, &newTexture.channels, 4);
	newTexture.name = filePath;

    if (!newTexture.data){
		cout << "File cannot be found: " << filePath << endl;
        newTexture.data  = stbi_load("assets/images/emptyTexture.png", &newTexture.width, &newTexture.height, &newTexture.channels, 4);
        newTexture.name = "assets/images/emptyTexture.png";
    }
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, newTexture.textureId);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, newTexture.width,
                 newTexture.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, newTexture.data);
    glGenerateMipmap(GL_TEXTURE_2D);
	
	return newTexture;
}
texture emptyTexture;
void enableTexture(texture usedTexture){
	glBindTexture(GL_TEXTURE_2D, usedTexture.textureId);
}                  
