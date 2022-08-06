#pragma once
#ifndef INTERFACE_H
#define INTERFACE_H

#include <glew.h>
#include <glfw3.h>

#include "graphics.h"
#include "texture.h"

#include <vector>
#include <iostream>
#include <map>
using namespace std;

#include <glm.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace glm;

#include <ft2build.h>
#include FT_FREETYPE_H

#define DEFAULT_FONT_SIZE 11

template<class T>
int newVectorPos(vector<T> * used) {
	int size = used->size();
	used->resize(size + 1);
	return size;
}

//buttons
class button {
public:
	string name = "New Button";

	vec3 position = vec3(0.0f, 0.0f, 0.0f);
	vec2 scale = vec2(1.0f, 1.0f);
	float rotation = 0.0f;

	bool clickUp = false;
	bool clickDown = false;

	bool interactive = true;
	bool mouseOver = false;

	int minX = 0;
	int maxX = 0;
	int minY = 0;
	int maxY = 0;

	texture texture;

	vec3 colour = vec3(1.0f, 1.0f, 1.0f);
	float alpha = 1.0f;

	bool active = true;
};

extern vector<button> allButtons;
int createButton(vec2 size = vec2(1.0f), vec3 position = vec3(1.0f), bool interactive = true); // return position in allbuttons vector
void renderButtons(); //bind vertexarray and draw with texture

extern double mousePosX, mousePosY;
void updateMousePos();
void registerClicks(); //lots of maths

void buttonsBegin(); //start button VBO, VAO & EBO and reserve in memory

void interfaceMainloop();
void interfaceBegin();

#endif
