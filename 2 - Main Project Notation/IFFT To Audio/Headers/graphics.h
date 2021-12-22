#ifndef GRAPHICS_H
#define GRAPHICS_H

#include <iostream>
#include <map>
using namespace std;

#include <ft2build.h>
#include FT_FREETYPE_H

#include "filesShaders.h"

extern GLFWwindow* window;
bool startOpenGL(GLFWwindow*& used_window, int width, int height);

struct Character {
	GLuint TextureID;   // ID handle of the glyph texture
	ivec2 Size;    // Size of glyph
	ivec2 Bearing;  // Offset from baseline to left/top of glyph
	GLuint Advance;    // Horizontal offset to advance to next glyph
};
extern map<GLchar, Character> fontCharacters;

#define LOADED_FONT_SIZE 25
map<GLchar, Character> getFont(const char* path, int fontsize);
void textsBegin();

vec2 renderText(string displayedText, vec2 position, float alpha, float size, vec3 colour, map<GLchar, Character> Characters);
#endif // !GRAPHICS_H
