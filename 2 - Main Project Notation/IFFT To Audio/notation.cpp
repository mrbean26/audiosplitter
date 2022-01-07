#include "Headers/notation.h"

unsigned int notationShader;
unsigned int imageShader;

GLuint trebleClefTexture;
GLuint quarterNoteTexture;

vector<GLuint> notationSizes;
vector<GLuint> notationVAOs;
vector<GLuint> notationVBOs;

vec2 notationNoteSize;

void notationBegin() {
	// Begin Shader
	int vertShader = createShader("Assets/Shaders/tabVert.txt", GL_VERTEX_SHADER);
	int fragShader = createShader("Assets/Shaders/tabFrag.txt", GL_FRAGMENT_SHADER);
	notationShader = createProgram({ vertShader, fragShader });

	// Create Stave
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

	// Bar Line
	notationVAOs.push_back(0); notationVBOs.push_back(0);
	vertices = {
		0.0f, fdisplay_y,
		0.0f, display_y - 4.0f * NOTATION_LINE_GAP * display_y // Stave Height 
	};

	GLuint barLineSize = readyVertices(&notationVAOs[2], &notationVBOs[2], vertices, 2);
	notationSizes.push_back(barLineSize);

	// Quarter Note Texture & Coordinates
	quarterNoteTexture = readyTexture("Assets/quarterNote.png");
	float noteSize = NOTATION_LINE_GAP;

	startX = 0.0f;
	finalX = (1.294f * (fdisplay_y / fdisplay_x) * noteSize) * display_x;

	startY = display_y;
	finalY = display_y - noteSize * display_y;

	notationNoteSize = vec2(finalX, noteSize * display_y);

	vector<float> noteVertices = {
		finalX, startY, 1.0f, 1.0f, // Triangle Tex Coords
		finalX, finalY, 1.0f, 0.0f,
		startX, startY, 0.0f, 1.0f,

		finalX, finalY, 1.0f, 0.0f,
		startX, finalY, 0.0f, 0.0f,
		startX, startY, 0.0f, 1.0f
	};

	notationVAOs.push_back(0); notationVBOs.push_back(0);
	notationSizes.push_back(noteVertices.size() / 4);

	glGenVertexArrays(1, &notationVAOs[3]);
	glGenBuffers(1, &notationVBOs[3]);

	glBindVertexArray(notationVAOs[3]);
	glBindBuffer(GL_ARRAY_BUFFER, notationVBOs[3]);

	glBufferData(GL_ARRAY_BUFFER, noteVertices.size() * sizeof(float), noteVertices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	// Note Line
	startX = display_y;
	finalY = display_y - staveHeight * display_y;

	startX = 0.0f;
	finalX = NOTATION_NOTE_LINE_WIDTH * display_x;

	vector<float> noteLineVertices = {
		finalX, startY,
		finalX, finalY,
		startX, startY,

		finalX, finalY,
		startX, finalY,
		startX, startY
	};

	notationVAOs.push_back(0); notationVBOs.push_back(0);
	GLuint noteLineSize = readyVertices(&notationVAOs[4], &notationVBOs[0], noteLineVertices, 2);

	notationSizes.push_back(noteLineSize);
}

vector<bool> findKey(vector<vector<int>> notes) {
	vector<int> resultantNotes(5); // 0 = not found, 1 = not sharp, 2 = sharp

	// Look through notes
	int chunkCount = notes.size();
	
	for (int i = 0; i < chunkCount; i++) {
		int noteCount = notes[i].size();

		for (int j = 0; j < noteCount; j++) {
			int currentNote = notes[i][j];
			int octaveNote = currentNote % 12;
			
			if (octaveNote == 0) {
				resultantNotes[0] = 1; // A
			}
			if (octaveNote == 1 && resultantNotes[0] == 0) {
				resultantNotes[0] = 2;
			}

			if (octaveNote == 3) {
				resultantNotes[1] = 1; // C
			}
			if (octaveNote == 4 && resultantNotes[1] == 0) {
				resultantNotes[1] = 2;
			}

			if (octaveNote == 5) {
				resultantNotes[2] = 1; // D
			}
			if (octaveNote == 6 && resultantNotes[2] == 0) {
				resultantNotes[2] = 2;
			}

			if (octaveNote == 8) {
				resultantNotes[3] = 1; // F
			}
			if (octaveNote == 9 && resultantNotes[3] == 0) {
				resultantNotes[3] = 2;
			}

			if (octaveNote == 10) {
				resultantNotes[4] = 1; // G
			}
			if (octaveNote == 11 && resultantNotes[4] == 0) {
				resultantNotes[4] = 2;
			}
		}
	}

	// A true variable means that the natural note (eg A, C, D) is shifted (eg A#, C#, D#)
	vector<bool> result(5);

	for (int i = 0; i < 5; i++) {
		if (resultantNotes[i] == 2) {
			result[i] = true;
		}
		else {
			result[i] = false;
		}
		if (i > 3) {
			result[i] = true;
		}
	}
	return result;
}
void drawKeySignature(vector<bool> keySignature, float yOffset) {
	vector<int> keyDistances = { 0, 2, 3, 5, 6 };

	float staveHeight = 4.0f * NOTATION_LINE_GAP;
	float fdisplay_x = float(display_x);
	float fdisplay_y = float(display_y);

	float trebleClefWidth = (NOTATION_EDGE_DISTANCE + staveHeight * 0.587f * 0.587f * (fdisplay_y / fdisplay_x)) * display_x;
	int count = 1;

	float textSize = NOTATION_SHARP_SIZE * (fdisplay_y / 1000.0f);

	renderText("", vec2(320.0f, 180.0f), 1.0f, textSize, vec3(0.0f), fontCharacters); // For some reason they only render when this is here
	for (int i = 0; i < 5; i++) {
		if (keySignature[i]) {
			float xPosition = trebleClefWidth + count * NOTATION_SHARP_DISTANCE * display_x;
			float yPosition = (1.0f + yOffset) * display_y - (3.0f * NOTATION_LINE_GAP * display_y); // Position of "A" Note
			yPosition = yPosition + display_y * keyDistances[i] * NOTATION_LINE_GAP * 0.5f;
			
			renderText("#", vec2(xPosition, yPosition), 1.0f, textSize, vec3(0.0f), fontCharacters);
			count = count + 1;
		}
	}

	// Bar Line After the clef & signature
	drawBarLine(trebleClefWidth + (count + 1) * NOTATION_SHARP_DISTANCE * display_x, yOffset * display_y);
}

void drawTrebleClef(float yOffset) {
	glUseProgram(imageShader);

	glBindVertexArray(notationVAOs[1]);
	glBindTexture(GL_TEXTURE_2D, trebleClefTexture);

	mat4 scalePositionMatrix = ortho(0.0f, static_cast<GLfloat>(display_x), 0.0f, static_cast<GLfloat>(display_y));
	scalePositionMatrix = translate(scalePositionMatrix, vec3(0.0f, yOffset, 0.0f));

	setMat4(imageShader, "scalePositionMatrix", scalePositionMatrix);
	glDrawArrays(GL_TRIANGLES, 0, notationSizes[1]);
}
void drawBarLine(float xOffset, float yOffset) {
	glUseProgram(notationShader);

	mat4 projectionMatrix = ortho(0.0f, static_cast<GLfloat>(display_x), 0.0f, static_cast<GLfloat>(display_y));
	projectionMatrix = translate(projectionMatrix, vec3(xOffset, yOffset, 0.0f));
	setMat4(notationShader, "projection", projectionMatrix);

	glBindVertexArray(notationVAOs[2]);
	glDrawArrays(GL_LINES, 0, notationSizes[2]);
}

void drawSingularNote(vec2 noteRootPosition, float staveCenter) {
	// Note Circle
	glUseProgram(imageShader);
	mat4 projectionMatrix = ortho(0.0f, static_cast<GLfloat>(display_x), 0.0f, static_cast<GLfloat>(display_y));

	glBindVertexArray(notationVAOs[3]);
	glBindTexture(GL_TEXTURE_2D, quarterNoteTexture);

	mat4 scalePositionMatrix = translate(projectionMatrix, vec3(noteRootPosition.x * display_x, noteRootPosition.y * display_y, 0.0f));
	setMat4(imageShader, "scalePositionMatrix", scalePositionMatrix);

	glDrawArrays(GL_TRIANGLES, 0, notationSizes[3]);

	// Draw Line
	glUseProgram(notationShader);
	float staveHeight = 4.0f * NOTATION_LINE_GAP;
	vec2 linePosition = vec2(noteRootPosition.x * display_x, noteRootPosition.y * display_y);

	if (noteRootPosition.y * display_y <= staveCenter * display_y) {
		// Put Line Upwards
		linePosition.x += notationNoteSize.x - NOTATION_NOTE_LINE_WIDTH * display_x;
		linePosition.y += staveHeight * display_y - notationNoteSize.y / 2.0f;
	}
	else {
		// Put Line Downwards
		linePosition.x += NOTATION_NOTE_LINE_WIDTH * display_x;
		linePosition.y -= notationNoteSize.y / 2.0f;
	}
	
	projectionMatrix = translate(projectionMatrix, vec3(linePosition, 0.0f));
	setMat4(notationShader, "projection", projectionMatrix);

	glBindVertexArray(notationVAOs[4]);
	glDrawArrays(GL_TRIANGLES, 0, notationSizes[4]);
}
void drawNotes(vector<vector<int>> notes, vector<bool> keySignature) {
	// Calculate Distance Up To First Note
	float fdisplay_x = float(display_x);
	float fdisplay_y = float(display_y);

	// Treble Clef Width
	float staveHeight = 4.0f * NOTATION_LINE_GAP;
	float trebleClefWidth = staveHeight * 0.587f * 0.587f * (fdisplay_y / fdisplay_x);

	// Key Signature Width
	int signatureCount = 2;

	int signatureSize = keySignature.size();
	for (int i = 0; i < signatureSize; i++) {
		if (keySignature[i]) {
			signatureCount = signatureCount + 1;
		}
	}

	float keySignatureWidth = NOTATION_SHARP_DISTANCE * signatureCount;

	// Draw Notes
	int currentStave = 0;
	float currentYPosition = -NOTATION_EDGE_DISTANCE - NOTATION_MAX_LEDGER_LINES * NOTATION_LINE_GAP;
	float initialNoteXPosition = NOTATION_EDGE_DISTANCE + trebleClefWidth + keySignatureWidth;

	int noteCount = notes.size();
	int notesPerLine = NOTATION_CHUNKS_PER_LINE * (fdisplay_x / 1000.0f);
	float noteGapDistance = (1.0f - NOTATION_EDGE_DISTANCE - initialNoteXPosition) / float(notesPerLine);

	for (int i = 0; i < noteCount; i++) {
		int noteStaveIndex = i % notesPerLine;
		float currentXPosition = initialNoteXPosition + noteGapDistance * noteStaveIndex;
		
		int subNoteCount = notes[i].size();
		for (int j = 0; j < subNoteCount; j++) {
			// Current Y Position is Top E (Top Stave Line) - Note 31
			// Shift Accordingly to note

			int noteDistance = notes[i][j] - 31; // 31 is the start position note ("E")
			float shiftedYPosition = noteDistance * 0.25f * NOTATION_LINE_GAP;
			float finalYPosition = currentYPosition + shiftedYPosition;
			
			drawSingularNote(vec2(currentXPosition, finalYPosition), currentYPosition - NOTATION_LINE_GAP * 2.0f);
		}

		// Draw Bar Lines
		if (i % 4 == 0 && i > 0) {
			float barLineXPosition = currentXPosition;
			drawBarLine(barLineXPosition * display_x, currentYPosition * display_y);

			glUseProgram(imageShader);

			glBindVertexArray(notationVAOs[3]);
			glBindTexture(GL_TEXTURE_2D, quarterNoteTexture);
		}

		if (noteStaveIndex == notesPerLine - 1) {
			currentStave = currentStave + 1;
			currentYPosition = currentYPosition - (5 * NOTATION_LINE_GAP + NOTATION_EDGE_DISTANCE + 2 * NOTATION_MAX_LEDGER_LINES * NOTATION_LINE_GAP);
		}
	}
}

void drawStaveLines(float yOffset) {
	glUseProgram(notationShader);

	mat4 projectionMatrix = ortho(0.0f, static_cast<GLfloat>(display_x), 0.0f, static_cast<GLfloat>(display_y));
	projectionMatrix = translate(projectionMatrix, vec3(0.0f, yOffset, 0.0f));
	setMat4(notationShader, "projection", projectionMatrix);

	glBindVertexArray(notationVAOs[0]);
	glDrawArrays(GL_LINES, 0, notationSizes[0]);
}
void drawNotation(vector<vector<int>> notes, vector<bool> keySignature) {
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
		drawTrebleClef(yOffset * display_y);

		// Draw Key Signature
		drawKeySignature(keySignature, yOffset);
	}

	// Draw Notes
	drawNotes(notes, keySignature);
}