#include "Headers/graphics.h"

GLFWwindow* window;
int display_x, display_y;

bool startOpenGL(GLFWwindow*& used_window, int width, int height) {
	if (!glfwInit()) {
		return false;
	}

	display_x = width;
	display_y = height;

	GLFWmonitor* primary_monitor = glfwGetPrimaryMonitor();
	const GLFWvidmode* monitor_mode = glfwGetVideoMode(primary_monitor);
	int monitor_x = monitor_mode->width;
	int monitor_y = monitor_mode->height;

	// Stop Resizing
	glewExperimental = GL_TRUE;
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	// shader for mac
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
	//used variables
	used_window = glfwCreateWindow(width, height, "Notation Viewer", NULL, NULL);
	glfwMakeContextCurrent(used_window);

	if (glewInit() != GLEW_OK) {
		return false;
	}
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glfwSwapInterval(0); // Unlimited Frames

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_DEPTH_TEST);

	return true;
}

map<GLchar, Character> fontCharacters;
unsigned int textShader;
GLuint textVAO, textVBO;

map<GLchar, Character> getFont(const char* path, int fontsize) {
	map<GLchar, Character> returned;

	FT_Library ftLibrary;
	if (FT_Init_FreeType(&ftLibrary)) {
		cout << "Couldn't Init Freetype" << endl;
		return returned;
	}
	FT_Face newFont;
	if (FT_New_Face(ftLibrary, path, 0, &newFont)) {
		cout << "Couldn't load font: " << path << endl;
		return returned;
	}
	FT_Set_Pixel_Sizes(newFont, 0, fontsize);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	for (GLubyte c = 0; c < 128; c++) { //load first 128 of ASCII
		if (FT_Load_Char(newFont, c, FT_LOAD_RENDER)) {
			cout << "Couldn't Load Character" << c << endl;
			continue;
		}
		// Generate texture
		GLuint texture;
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexImage2D(
			GL_TEXTURE_2D, 0, GL_RED,
			newFont->glyph->bitmap.width,
			newFont->glyph->bitmap.rows,
			0, GL_RED, GL_UNSIGNED_BYTE,
			newFont->glyph->bitmap.buffer
		);
		// Set texture options
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		// Now store character for later use
		Character character = {
			texture,
			glm::ivec2(newFont->glyph->bitmap.width, newFont->glyph->bitmap.rows),
			glm::ivec2(newFont->glyph->bitmap_left, newFont->glyph->bitmap_top),
			(GLuint)newFont->glyph->advance.x
		};
		returned.insert(std::pair<GLchar, Character>(c, character));
	}
	FT_Done_Face(newFont);
	FT_Done_FreeType(ftLibrary);
	return returned;
}
void textsBegin() {
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	int vertShader = createShader("Assets/Shaders/textVert.txt", GL_VERTEX_SHADER);
	int fragShader = createShader("Assets/Shaders/textFrag.txt", GL_FRAGMENT_SHADER);
	textShader = createProgram({ vertShader, fragShader });
	//load fonts
	fontCharacters = getFont("Assets/TabFont.ttf", LOADED_FONT_SIZE);
	glBindTexture(GL_TEXTURE_2D, 0);
	//ready vbo & vao
	glGenVertexArrays(1, &textVAO);
	glGenBuffers(1, &textVBO);
	glBindVertexArray(textVAO);
	glBindBuffer(GL_ARRAY_BUFFER, textVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

vec2 renderText(string displayedText, vec2 position, float alpha, float size, vec3 colour, map<GLchar, Character> Characters) {
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDepthFunc(GL_LEQUAL); // allows text over skybox

	mat4 projectionMatrix = ortho(0.0f, static_cast<GLfloat>(display_x), 0.0f, static_cast<GLfloat>(display_y));
	setMat4(textShader, "projection", projectionMatrix);

	glUseProgram(textShader);
	glActiveTexture(GL_TEXTURE0);
	int textureLocation = glGetUniformLocation(textShader, "text");
	glUniform1i(textureLocation, 0);
	int colourLocation = glGetUniformLocation(textShader, "textColor");
	glUniform4f(colourLocation, colour.x, colour.y, colour.z, alpha);

	glBindVertexArray(textVAO);

	float totalWidth = 0.0f;
	float totalHeight = 0.0f;

	// Iterate through all characters
	std::string::const_iterator c;
	for (c = displayedText.begin(); c != displayedText.end(); c++)
	{
		Character ch = Characters[*c];

		GLfloat xpos = position.x + ch.Bearing.x * size;
		GLfloat ypos = position.y - (ch.Size.y - ch.Bearing.y) * size;

		GLfloat w = ch.Size.x * size;
		GLfloat h = ch.Size.y * size;

		totalWidth += ((ch.Advance >> 6) * size) + ((ch.Size.x >> 6) * size);
		if (h > totalHeight) {
			totalHeight = h;
		}

		// Update VBO for each character
		GLfloat vertices[6][4] = {
			{ xpos,     ypos + h,   0.0, 0.0 },
			{ xpos,     ypos,       0.0, 1.0 },
			{ xpos + w, ypos,       1.0, 1.0 },

			{ xpos,     ypos + h,   0.0, 0.0 },
			{ xpos + w, ypos,       1.0, 1.0 },
			{ xpos + w, ypos + h,   1.0, 0.0 }
		};
		// Render glyph texture over quad
		glBindTexture(GL_TEXTURE_2D, ch.TextureID);
		// Update content of VBO memory
		glBindBuffer(GL_ARRAY_BUFFER, textVBO);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices); // Be sure to use glBufferSubData and not glBufferData

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		// Render quad
		glDrawArrays(GL_TRIANGLES, 0, 6);
		// Now advance cursors for next glyph (note that advance is number of 1/64 pixels)
		position.x += (ch.Advance >> 6) * size; // Bitshift by 6 to get value in pixels (2^6 = 64 (divide amount of 1/64th pixels by 64 to get amount of pixels))
	}
	glBindVertexArray(0);
	glBindTexture(GL_TEXTURE_2D, 0);
	return vec2(totalWidth, totalHeight);
}