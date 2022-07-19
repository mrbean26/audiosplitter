#pragma once
#ifndef TEXTURE_H
#define TEXTURE_H

#include "stb_image.h" //image library

class texture { //holds all data
public:
	unsigned int textureId;
	int width, height;
	int channels;
	unsigned char * data;
	const char * name;
};

texture loadTexture(const char * filePath); // load texture from file and return a relevant texture class
extern texture emptyTexture;
void enableTexture(texture usedTexture); // ready for drawing shapes

#endif
