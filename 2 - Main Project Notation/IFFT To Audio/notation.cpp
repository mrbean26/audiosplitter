#include "Headers/notation.h"

unsigned int notationShader;
unsigned int imageShader;
GLuint trebleClefTexture;

vector<GLuint> notationSizes;
vector<GLuint> notationVAOs;
vector<GLuint> notationVBOs;

void notationBegin() {
	// Begin Shader
	int vertShader = createShader("Assets/Shaders/tabVert.txt", GL_VERTEX_SHADER);
	int fragShader = createShader("Assets/Shaders/tabFrag.txt", GL_FRAGMENT_SHADER);
	notationShader = createProgram({ vertShader, fragShader });

	// Create Stave
	glLineWidth(10.0f);

	notationVAOs.push_back(0); notationVBOs.push_back(0);
	vector<float> vertices = {};

	for (int i = 0; i < 5; i++) { // 5 lines due to 5 lines on a stave
		vertices.push_back(display_x * NOTATION_EDGE_DISTANCE);
		vertices.push_back(display_y - (i * NOTATION_LINE_GAP * display_y));

		vertices.push_back(display_x - display_x * NOTATION_EDGE_DISTANCE);
		vertices.push_back(display_y - (i * NOTATION_LINE_GAP * display_y));
	}

	GLuint staveSize = readyVertices(&notationVAOs[0], &notationVBOs[0], vertices, 2);
	notationSizes.push_back(staveSize);

	// Treble Clef
	trebleClefTexture = readyTexture("Assets/trebleClef.png");

	float fdisplay_x = float(display_x);
	float fdisplay_y = float(display_y);

	// Treble Clef Coordinates
	float staveHeight = 4.0f * NOTATION_LINE_GAP;

	float startX = NOTATION_EDGE_DISTANCE * display_x;
	float finalX = (NOTATION_EDGE_DISTANCE + staveHeight * 0.587f * 0.587f * (fdisplay_y / fdisplay_x)) * display_x;

	float startY = display_y;
	float finalY = display_y - staveHeight * display_y;

	vector<float> trebleClefVertices = {
		finalX, startY, 1.0f, 1.0f, // Triangle Tex Coords
		finalX, finalY, 1.0f, 0.0f,
		startX, startY, 0.0f, 1.0f,

		finalX, finalY, 1.0f, 0.0f,
		startX, finalY, 0.0f, 0.0f,
		startX, startY, 0.0f, 1.0f
	};

	notationVAOs.push_back(0); notationVBOs.push_back(0);
	notationSizes.push_back(trebleClefVertices.size() / 4);

	glGenVertexArrays(1, &notationVAOs[1]);
	glGenBuffers(1, &notationVBOs[1]);

	glBindVertexArray(notationVAOs[1]);
	glBindBuffer(GL_ARRAY_BUFFER, notationVBOs[1]);

	glBufferData(GL_ARRAY_BUFFER, trebleClefVertices.size() * sizeof(float), trebleClefVertices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
	
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	vertShader = createShader("Assets/Shaders/textureVert.txt", GL_VERTEX_SHADER);
	fragShader = createShader("Assets/Shaders/textureFrag.txt", GL_FRAGMENT_SHADER);
	imageShader = createProgram({ vertShader, fragShader });
}

void drawStaveLines(float yOffset) {
	glUseProgram(notationShader);

	mat4 projectionMatrix = ortho(0.0f, static_cast<GLfloat>(display_x), 0.0f, static_cast<GLfloat>(display_y));
	projectionMatrix = translate(projectionMatrix, vec3(0.0f, yOffset, 0.0f));
	setMat4(notationShader, "projection", projectionMatrix);

	glBindVertexArray(notationVAOs[0]);
	glDrawArrays(GL_LINES, 0, notationSizes[0]);
}
void drawNotation(vector<vector<int>> notes) {
	int chunkCount = notes.size();
	int chunksPerLine = NOTATION_CHUNKS_PER_LINE * (double(display_x) / 1000.0);
	int requiredStaves = ceil(double(chunkCount) / chunksPerLine);

	// Draw Staves
	for (int i = 0; i < requiredStaves; i++) {
		float yOffset = -NOTATION_EDGE_DISTANCE;
		yOffset = yOffset - NOTATION_MAX_LEDGER_LINES * NOTATION_LINE_GAP; // Top ledger lines of first stave

		yOffset = yOffset - i * (5 * NOTATION_LINE_GAP + NOTATION_EDGE_DISTANCE);
		yOffset = yOffset - i * (2 * NOTATION_MAX_LEDGER_LINES * NOTATION_LINE_GAP); // Bottom and top set of ledger lines 
		drawStaveLines(yOffset * display_y);

		// Draw Treble Clefs
		glUseProgram(imageShader);

		glBindVertexArray(notationVAOs[1]);
		glBindTexture(GL_TEXTURE_2D, trebleClefTexture);

		mat4 scalePositionMatrix = ortho(0.0f, static_cast<GLfloat>(display_x), 0.0f, static_cast<GLfloat>(display_y));
		scalePositionMatrix = translate(scalePositionMatrix, vec3(0.0f, yOffset * display_y, 0.0f));
		
		setMat4(imageShader, "scalePositionMatrix", scalePositionMatrix);
		glDrawArrays(GL_TRIANGLES, 0, notationSizes[1]);
	}
}