#include "Headers/interface.h"

#include <vector>
using namespace std;

unsigned int buttonTextureShader;
bool clickedDown = false;
bool lastClick = false;

GLuint buttonVAO, buttonVBO, buttonEBO;

std::vector<button> allButtons;
int createButton(vec2 size, vec3 position, bool interactive){
	button newButton;
	allButtons[newVectorPos(&allButtons)] = newButton;
	int index = allButtons.size() - 1;
	allButtons[index].scale = size;
	allButtons[index].position = position;
	allButtons[index].interactive = interactive;
	return index;
}

void renderButtons(){
	glUseProgram(buttonTextureShader);

	glDisable(GL_DEPTH_TEST);
	int buttonCount = allButtons.size();
	vec2 rescale = vec2(2.0f, 2.0f);
	vec2 aspectRatio = vec2(aspect_x, aspect_y);
	glBindVertexArray(buttonVAO);
	for (int i = 0; i < buttonCount; i++) {
		button currentButton = allButtons[i];
		if (!currentButton.active) {
			continue;
		}
		vec2 scale = currentButton.scale * rescale;
		vec3 position = currentButton.position;
		scale = scale / aspectRatio;
		//make button bigger if mouse is over it
		mat4 scaleMat = mat4(1.0f);
		if (currentButton.mouseOver && currentButton.interactive) {
			scale *= vec2(1.05f, 1.05f);
		}
		if (currentButton.clickDown && currentButton.interactive) {
			scale *= vec2(0.95f, 0.95f);
		}
		//rescale the matrix and send position info to shader
		scaleMat = translate(scaleMat, position);
		scaleMat = glm::scale(scaleMat, vec3(scale, 1.0f));
		scaleMat = glm::rotate(scaleMat, radians(currentButton.rotation), vec3(0.0f, 0.0f, 1.0f));
		//update position, scale and rotation info ready for the shader to use
		setMat4(buttonTextureShader, "scalePositionMatrix", scaleMat);
		setVec3(buttonTextureShader, "imageColor", currentButton.colour);

		//draw
        enableTexture(currentButton.texture);
		//set texture for shader

		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0); //if an error is being shown here for memory, shapes are being created before backendBegin()
	}
	glEnable(GL_DEPTH_TEST);
}

void registerClicks(){
	int mouseState = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
	if (mouseState == GLFW_PRESS) {
		clickedDown = true;
	}
	else {
		clickedDown = false;
	}
	//reset
	int buttonCount = allButtons.size();
	for (int i = 0; i < buttonCount; i++) {
		allButtons[i].clickUp = false;
	}
	//mouse pos
	for (int i = 0; i < buttonCount; i++) {
		if (!allButtons[i].active || !allButtons[i].interactive) {
			continue;
		}
		allButtons[i].clickDown = false;
		//button
		button currentButton = allButtons[i];
		vec3 buttonPosition = currentButton.position;
		vec2 buttonScale = currentButton.scale;
		//vertex coords
		int minX = 0, maxX = 0;
		int minY = 0, maxY = 0;
		//variables required to calculate minimum and maximum mouse positions for buttons to interact
		float midX = (display_x / 2.0f) * (1.0f + allButtons[i].position.x);
		float midY = display_y - (display_y / 2.0f) * (1.0f + allButtons[i].position.y);

		float xDivided = (float)display_x / (float)aspect_x;
		float yDivided = (float)display_y / (float)aspect_y;

		minX = midX - xDivided * allButtons[i].scale.x;
		maxX = midX + xDivided * allButtons[i].scale.x;

		minY = midY - yDivided * allButtons[i].scale.y;
		maxY = midY + yDivided * allButtons[i].scale.y;

		//add to class
		allButtons[i].minX = minX;
		allButtons[i].maxX = maxX;
		allButtons[i].minY = minY;
		allButtons[i].maxY = maxY;
		//check for click
		allButtons[i].mouseOver = false;
		if (mousePosX >= minX && mousePosX <= maxX) {
			if (mousePosY >= minY && mousePosY <= maxY) {
				if (clickedDown) {
					allButtons[i].clickDown = true;
				}
				if (lastClick && !clickedDown) {
					allButtons[i].clickUp = true;
				}
				allButtons[i].mouseOver = true;
			}
		}
	}
	lastClick = clickedDown;
}

void buttonsBegin(){
	//vertex data
	float vertices[] = {
		// positions then colors then texture coords, they are 0.99f to stop weird lines on edge of textures
		0.99f, 0.99f, 0.0f,
		 0.99f, 0.99f, 0.0f, 0.99f, 0.99f,
		 0.99f, -0.99f, 0.0f,
		 0.99f, 0.0f, 0.0f, 0.99f, 0.0f,
		 -0.99f, -0.99f, 0.0f,
		 0.0f, 0.0f, 0.99f, 0.0f, 0.0f,
		 -0.99f, 0.99f, 0.0f,
		 0.0f, 0.99f, 0.0f, 0.0f, 0.99f
	};
	unsigned int indices[] = {
		0, 1, 3, // first triangle
		1, 2, 3  // second triangle
	};
	glGenVertexArrays(1, &buttonVAO);
	glGenBuffers(1, &buttonVBO);
	glGenBuffers(1, &buttonEBO);
	glBindVertexArray(buttonVAO);
	glBindBuffer(GL_ARRAY_BUFFER, buttonVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buttonEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	// texture coord attribute
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);
	//create shader
	int vertexShader, fragmentShader;
	vertexShader = createShader("assets/shaders/textureVert.txt", GL_VERTEX_SHADER);
	fragmentShader = createShader("assets/shaders/textureFrag.txt", GL_FRAGMENT_SHADER);
	buttonTextureShader = createProgram({ vertexShader, fragmentShader });
}

void interfaceBegin(){
	buttonsBegin();
}

void interfaceMainloop(){
	renderButtons();
	updateMousePos();
	registerClicks();
}

void interfaceLastcall(){
	renderButtons();
}

double mousePosX, mousePosY;
void updateMousePos() {
	glfwGetCursorPos(window, &mousePosX, &mousePosY);
}

bool checkKey(int key){
	vector<int> glfwMouse = { GLFW_MOUSE_BUTTON_RIGHT, GLFW_MOUSE_BUTTON_LEFT, GLFW_MOUSE_BUTTON_MIDDLE };
	vector<int> blitzMouse = { 256256, 128128, 512512 };
	for (int i = 0; i < 3; i++) {
		if (key == blitzMouse[i]) {
			int state = glfwGetMouseButton(window, glfwMouse[i]);
			if (state == GLFW_PRESS) {
				return true;
			}
			return false;
		}
	}
	int keyState = glfwGetKey(window, key);
	if (keyState == GLFW_PRESS) {
		return true;
	}
	return false;
}

vector<bool> allKeysPrevious(146);
vector<bool> mousePrevious = { false, false, false };
vector<int> keyIndexes = { 32, 39, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
							69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 96, 
							161, 162, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 
							274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 
							294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 
							314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 
							334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347 };
bool checkKeyDown(int key) {
	vector<int> blitzMouse = { 256256, 128128, 512512 };
	bool returned = false;
	for (int m = 0; m < 3; m++) {
		if (key == blitzMouse[m]) {
			if (checkKey(blitzMouse[m])) {
				if (!mousePrevious[m]) {
					returned = true;
				}
				continue;
			}
		}
	}
	if (key == 256256 || key == 128128 || key == 512512) {
		return returned;
	}
	// keys
	if (checkKey(key)) {
		vector<int>::iterator iterator = find(keyIndexes.begin(), keyIndexes.end(), key);
		int index = std::distance(keyIndexes.begin(), iterator);
		if (!allKeysPrevious[index]) {
			return true;
		}
	}
	return false;
}

// all glfw keys
void updateKeys() {
	for (int i = 0; i < 146; i++) {
		allKeysPrevious[i] = false;
		if (checkKey(keyIndexes[i])) {
			allKeysPrevious[i] = true;
		}
	}
	vector<int> blitzMouse = { 256256, 128128, 512512 };
	for (int m = 0; m < 3; m++) {
		mousePrevious[m] = false;
		if (checkKey(blitzMouse[m])) {
			mousePrevious[m] = true;
		}
	}
}
