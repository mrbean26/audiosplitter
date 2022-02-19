#include "Headers/notation.h"
#include "Headers/audio.h"

notationViewer::notationViewer() {

}
notationViewer::notationViewer(vector<vector<int>> notes, int samplesPerChunk, int sampleRate, audioObject* trackAudio) {
	notationBegin();

	vector<vector<int>> newNotes = removeNoteRepetitions(notes);
	this->noteLengths = findNoteLengths(newNotes);

	this->keySignature = findKey(notes);

	samplesPerChunkProgress = samplesPerChunk;
	sampleRateProgress = sampleRate;

	trackObjectPointer = trackAudio;
}

void notationViewer::startNotationShaders() {
	// Tab Shaders
	int vertShader = createShader("Assets/Shaders/tabVert.txt", GL_VERTEX_SHADER);
	int fragShader = createShader("Assets/Shaders/tabFrag.txt", GL_FRAGMENT_SHADER);
	notationShader = createProgram({ vertShader, fragShader });

	// Image Shader
	vertShader = createShader("Assets/Shaders/textureVert.txt", GL_VERTEX_SHADER);
	fragShader = createShader("Assets/Shaders/textureFrag.txt", GL_FRAGMENT_SHADER);
	imageShader = createProgram({ vertShader, fragShader });
}
void notationViewer::notationBegin() {
	// Load & Initialise Shaders
	startNotationShaders();

	// Load Stave Lines into Memory
	startStaveLines();

	// Load Treble Clef into Memory
	startTrebleClef();

	// Load Bar Line into Memory
	startBarLine();

	// Load Notes & Each Texture into Memory
	startNotes();
	startNoteLine();

	startProgressBar();
}

void notationViewer::startTrebleClef() {
	// Treble Clef Texture
	trebleClefTexture = readyTexture("Assets/trebleClef.png");

	// Find Coordinates
	float startX = NOTATION_EDGE_DISTANCE * display_x; // Start of Stave X Coordinate
	float finalX = (NOTATION_EDGE_DISTANCE + TREBLE_CLEF_WIDTH * aspectRatioMultiplier) * display_x; // 0.3672 comes from image aspect ratio

	float startY = display_y; // Starts from the top (ready to be transformed)
	float finalY = display_y - STAVE_HEIGHT * display_y; // Height of the stave

	vector<float> trebleClefVertices = {
		finalX, startY, 1.0f, 1.0f, // Triangle Tex Coords
		finalX, finalY, 1.0f, 0.0f,
		startX, startY, 0.0f, 1.0f,

		finalX, finalY, 1.0f, 0.0f,
		startX, finalY, 0.0f, 0.0f,
		startX, startY, 0.0f, 1.0f
	};

	// Create OpenGL Attributes & Add to Vector
	notationVAOs.push_back(0); notationVBOs.push_back(0);
	notationSizes.push_back(trebleClefVertices.size() / 4);

	glGenVertexArrays(1, &notationVAOs[1]);
	glGenBuffers(1, &notationVBOs[1]);

	glBindVertexArray(notationVAOs[1]);
	glBindBuffer(GL_ARRAY_BUFFER, notationVBOs[1]);

	// Add data to buffer
	glBufferData(GL_ARRAY_BUFFER, trebleClefVertices.size() * sizeof(float), trebleClefVertices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
}
void notationViewer::drawTrebleClef(float yOffset) {
	glUseProgram(imageShader);

	glBindVertexArray(notationVAOs[1]);
	glBindTexture(GL_TEXTURE_2D, trebleClefTexture);

	mat4 scalePositionMatrix = ortho(0.0f, static_cast<GLfloat>(display_x), 0.0f, static_cast<GLfloat>(display_y));
	scalePositionMatrix = translate(scalePositionMatrix, vec3(0.0f, yOffset, 0.0f));

	setMat4(imageShader, "scalePositionMatrix", scalePositionMatrix * getViewMatrix());
	glDrawArrays(GL_TRIANGLES, 0, notationSizes[1]);
}

void notationViewer::drawLedgerLines(float noteY, float staveY, float noteX) {
	int requiredLedgerLinesUpper = (noteY - staveY) / NOTATION_LINE_GAP;
	for (int k = 0; k < requiredLedgerLinesUpper; k++) {
		float yPosition = staveY + NOTATION_LINE_GAP * (k + 1);
		drawLedgerLine(noteX * display_x, yPosition * display_y);
	}

	float bottomStave = staveY - 4.0f * NOTATION_LINE_GAP;
	int requiredLedgerLinesLower = (bottomStave - noteY) / NOTATION_LINE_GAP;
	for (int k = 0; k < requiredLedgerLinesLower; k++) {
		float yPosition = bottomStave - NOTATION_LINE_GAP * (k + 1);
		drawLedgerLine(noteX * display_x, yPosition * display_y);
	}
}
void notationViewer::drawLedgerLine(float xOffset, float yOffset) {
	glUseProgram(notationShader);

	// Find size to scale to note gap size (so that ledger line fits)
	int chunksPerLine = NOTATION_CHUNKS_PER_LINE * (double(display_x) / 1000.0);
	float length = 1.0f / float(chunksPerLine);

	mat4 projectionMatrix = ortho(0.0f, static_cast<GLfloat>(display_x), 0.0f, static_cast<GLfloat>(display_y));
	projectionMatrix = translate(projectionMatrix, vec3(xOffset, yOffset, 0.0f));
	projectionMatrix = scale(projectionMatrix, vec3(length, 1.0f, 1.0f));

	setMat4(notationShader, "projection", projectionMatrix * getViewMatrix());
	glBindVertexArray(notationVAOs[0]);
	glDrawArrays(GL_LINES, 0, 2);
}

void notationViewer::startStaveLines() {
	notationVAOs.push_back(0); notationVBOs.push_back(0);
	vector<float> vertices = {};

	for (int i = 0; i < 5; i++) { // 5 lines due to 5 lines on a stave
		vertices.push_back(display_x * NOTATION_EDGE_DISTANCE); // Starts from the left
		vertices.push_back(display_y - (i * NOTATION_LINE_GAP * display_y));

		vertices.push_back(display_x - display_x * NOTATION_EDGE_DISTANCE); // ends on the right
		vertices.push_back(display_y - (i * NOTATION_LINE_GAP * display_y));
	}

	GLuint staveSize = readyVertices(&notationVAOs[0], &notationVBOs[0], vertices, 2);
	notationSizes.push_back(staveSize);
}
void notationViewer::drawStaveLines(float yOffset) {
	glUseProgram(notationShader);

	mat4 projectionMatrix = ortho(0.0f, static_cast<GLfloat>(display_x), 0.0f, static_cast<GLfloat>(display_y));
	projectionMatrix = translate(projectionMatrix, vec3(0.0f, yOffset, 0.0f));
	setMat4(notationShader, "projection", projectionMatrix * getViewMatrix());

	glBindVertexArray(notationVAOs[0]);
	glDrawArrays(GL_LINES, 0, notationSizes[0]);
}

void notationViewer::startBarLine() {
	notationVAOs.push_back(0); notationVBOs.push_back(0);
	float fdisplay_x = float(display_x);
	float fdisplay_y = float(display_y);

	vector<float> vertices = {
		0.0f, fdisplay_y,
		0.0f, display_y - 4.0f * NOTATION_LINE_GAP * display_y // Stave Height 
	};

	GLuint barLineSize = readyVertices(&notationVAOs[2], &notationVBOs[2], vertices, 2);
	notationSizes.push_back(barLineSize);
}
void notationViewer::drawBarLine(float xOffset, float yOffset) {
	glUseProgram(notationShader);

	mat4 projectionMatrix = ortho(0.0f, static_cast<GLfloat>(display_x), 0.0f, static_cast<GLfloat>(display_y));
	projectionMatrix = translate(projectionMatrix, vec3(xOffset, yOffset, 0.0f));
	setMat4(notationShader, "projection", projectionMatrix * getViewMatrix());

	glBindVertexArray(notationVAOs[2]);
	glDrawArrays(GL_LINES, 0, notationSizes[2]);
}

void notationViewer::startNotes() {
	noteTextures.push_back(readyTexture("Assets/quarterNote.png"));
	noteTextures.push_back(readyTexture("Assets/halfNote.png"));
	noteTextures.push_back(readyTexture("Assets/threeQuarterNote.png"));
	noteTextures.push_back(readyTexture("Assets/halfNote.png"));

	// Note Coordinates
	float startX = 0.0f;
	float finalX = (NOTE_SIZE_WIDTH * aspectRatioMultiplier) * display_x; // 1.294 comes from image resolution ratio

	float startY = display_y;
	float finalY = display_y - NOTE_SIZE * display_y;

	// Declare Coordinates
	notationNoteSize = vec2(finalX, NOTE_SIZE * display_y);
	vector<float> noteVertices = {
		finalX, startY, 1.0f, 1.0f, // Triangle Coordinates: X, Y, TexX, TexY
		finalX, finalY, 1.0f, 0.0f,
		startX, startY, 0.0f, 1.0f,

		finalX, finalY, 1.0f, 0.0f,
		startX, finalY, 0.0f, 0.0f,
		startX, startY, 0.0f, 1.0f
	};

	// Start OpenGL Attributes
	notationVAOs.push_back(0); notationVBOs.push_back(0);
	notationSizes.push_back(noteVertices.size() / 4);

	glGenVertexArrays(1, &notationVAOs[3]);
	glGenBuffers(1, &notationVBOs[3]);

	glBindVertexArray(notationVAOs[3]);
	glBindBuffer(GL_ARRAY_BUFFER, notationVBOs[3]);

	// Load Data into Buffer
	glBufferData(GL_ARRAY_BUFFER, noteVertices.size() * sizeof(float), noteVertices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
}
void notationViewer::startNoteLine() {
	// Create line the height of the stave
	float startY = display_y;
	float finalY = display_y - STAVE_HEIGHT * display_y;

	float startX = 0.0f;
	float finalX = NOTATION_NOTE_LINE_WIDTH * display_x;

	// Declare Coordinates
	vector<float> noteLineVertices = {
		finalX, startY,
		finalX, finalY,
		startX, startY,

		finalX, finalY,
		startX, finalY,
		startX, startY
	};

	// Declare OpenGL Attributes
	notationVAOs.push_back(0); notationVBOs.push_back(0);
	GLuint noteLineSize = readyVertices(&notationVAOs[4], &notationVBOs[0], noteLineVertices, 2);

	notationSizes.push_back(noteLineSize);
}
void notationViewer::drawSingularNote(vec2 noteRootPosition, float staveCenter, int noteDuration, bool sharpSign) {
	// Draw Note Circle
	// Set Shader Details
	mat4 projectionMatrix = ortho(0.0f, static_cast<GLfloat>(display_x), 0.0f, static_cast<GLfloat>(display_y));
	mat4 scalePositionMatrix = translate(projectionMatrix, vec3(noteRootPosition.x * display_x, noteRootPosition.y * display_y, 0.0f));

	glUseProgram(imageShader);
	setMat4(imageShader, "scalePositionMatrix", scalePositionMatrix * getViewMatrix());

	// Draw Note
	glBindVertexArray(notationVAOs[3]);
	glBindTexture(GL_TEXTURE_2D, noteTextures[noteDuration - 1]);
	glDrawArrays(GL_TRIANGLES, 0, notationSizes[3]);

	if (noteDuration == 4) {
		return; // No line for full length note
	}

	// Draw Line
	vec2 linePosition = vec2(noteRootPosition.x * display_x, noteRootPosition.y * display_y);

	// Put Line Upwards if Note Center is Low Down, if not then do the oppposite
	if (noteRootPosition.y <= staveCenter) {
		// Put Line Upwards
		linePosition.x += notationNoteSize.x - NOTATION_NOTE_LINE_WIDTH * display_x;
		linePosition.y += STAVE_HEIGHT * display_y - notationNoteSize.y / 2.0f;
	}
	else {
		// Put Line Downwards
		linePosition.x += NOTATION_NOTE_LINE_WIDTH * display_x;
		linePosition.y -= notationNoteSize.y / 2.0f;
	}

	// Set Shader Details
	glUseProgram(notationShader);
	projectionMatrix = translate(projectionMatrix, vec3(linePosition, 0.0f));
	setMat4(notationShader, "projection", projectionMatrix * getViewMatrix());

	// Draw
	glBindVertexArray(notationVAOs[4]);
	glDrawArrays(GL_TRIANGLES, 0, notationSizes[4]);

	// Draw Sharp Sign
	if (sharpSign) {
		float textSize = NOTATION_SHARP_SIZE * (double(display_y) / 1000.0);

		noteRootPosition = vec2(noteRootPosition.x * display_x, display_y + noteRootPosition.y * display_y);
		// shift text for scrolling
		noteRootPosition.y += currentOffset;

		renderText("#", noteRootPosition, 1.0f, textSize, vec3(0.0f), fontCharacters);
	}
}

vector<bool> notationViewer::findKey(vector<vector<int>> notes) {
	vector<int> resultantNotes(5); // 0 = not found, 1 = not sharp, 2 = sharp

	// Look through notes
	int chunkCount = notes.size();
	
	for (int i = 0; i < chunkCount; i++) {
		int noteCount = notes[i].size();

		for (int j = 0; j < noteCount; j++) {
			int currentNote = notes[i][j];
			int octaveNote = currentNote % 12;
			
			// Check if first note is sharp or natural
			switch (octaveNote) {
			case 0:
				resultantNotes[0] = 1;
			case 1:
				if (resultantNotes[0] == 0) {
					resultantNotes[0] = 2;
				}
				break;
			case 3:
				resultantNotes[1] = 1;
			case 4:
				if (resultantNotes[1] == 0) {
					resultantNotes[1] = 2;
				}
				break;
			case 5:
				resultantNotes[2] = 1;
			case 6:
				if (resultantNotes[2] == 0) {
					resultantNotes[2] = 2;
				}
				break;
			case 8:
				resultantNotes[3] = 1;
			case 9:
				if (resultantNotes[3] == 0) {
					resultantNotes[3] = 2;
				}
				break;
			case 10:
				resultantNotes[4] = 1;
			case 11:
				if (resultantNotes[4] == 0) {
					resultantNotes[4] = 2;
				}
				break;
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
	}
	return result;
}
float notationViewer::getKeySignatureWidth() {
	int count = 2;

	int signatureSize = keySignature.size();
	for (int i = 0; i < signatureSize; i++) {
		if (keySignature[i]) {
			count = count + 1;
		}
	}

	return count * NOTATION_SHARP_DISTANCE;
}
void notationViewer::drawKeySignature(vector<bool> keySignature, float yOffset) {
	vector<int> keyDistances = { 0, 2, 3, 5, 6 };
	float trebleClefOffset = (NOTATION_EDGE_DISTANCE + TREBLE_CLEF_WIDTH * aspectRatioMultiplier) * display_x; // 0.3672 comes from image aspect ratio

	int count = 1; // initial offset away from treble clef
	float textSize = NOTATION_SHARP_SIZE * (float(display_y) / 1000.0f); // Size comes from a proportion of heights

	for (int i = 0; i < 5; i++) {
		if (keySignature[i]) {
			float xPosition = trebleClefOffset + count * NOTATION_SHARP_DISTANCE * display_x; // Shift along by how many sharps already present
			float yPosition = display_y + display_y * (yOffset - 3.0f * NOTATION_LINE_GAP); // shift to tab position then Position of "A" Note
			yPosition = yPosition + display_y * keyDistances[i] * NOTATION_LINE_GAP * 0.5f; // shift  to note position
			
			// shift text for scrolling
			yPosition += currentOffset;

			renderText("#", vec2(xPosition, yPosition), 1.0f, textSize, vec3(0.0f), fontCharacters);
			count = count + 1;
		}
	}

	// Bar Line After the clef & signature
	count = count + 1; // Offset distance from last sharp to the bar line
	drawBarLine(trebleClefOffset + count * NOTATION_SHARP_DISTANCE * display_x, yOffset * display_y);
}

bool notationViewer::compareNoteChunks(vector<int> chunkOne, vector<int> chunkTwo) {
	int size = chunkOne.size();
	int sizeTwo = chunkTwo.size();

	if (size != sizeTwo) {
		return false;
	}

	for (int i = 0; i < sizeTwo; i++) {
		if (chunkOne[i] != chunkTwo[i]) {
			return false;
		}
	}

	return true;
}
vector<vector<int>> notationViewer::removeNoteRepetitions(vector<vector<int>> originalChunks) {
	// Find Minimum Number of Consecutive Repetitions
	vector<int> currentChunk = originalChunks[0];
	int chunkCount = originalChunks.size();

	int currentCount = 0;
	int lowestCount = INT_MAX;

	for (int i = 0; i < chunkCount; i++) {
		if (compareNoteChunks(originalChunks[i], currentChunk)) {
			currentCount = currentCount + 1;
		}
		else {
			if (currentCount < lowestCount) {
				lowestCount = currentCount;
				currentCount = 0;
				currentChunk = originalChunks[i];
			}
		}
	}
	
	// Remove unneccesary repetitions
	vector<vector<int>> resultantChunks = { originalChunks[0] };
	currentChunk = originalChunks[0];

	for (int i = lowestCount; i < chunkCount; i++) {
		resultantChunks.push_back(originalChunks[i]);

		if (!compareNoteChunks(originalChunks[i], currentChunk)) {
			currentChunk = originalChunks[i];
			i = i + lowestCount; // Skip repeated chunks			
		}
	}

	return resultantChunks;
}
vector<vector<pair<int, int>>> notationViewer::findNoteLengths(vector<vector<int>> noteChunks) {
	vector<vector<int>> currentChunks = noteChunks;
	vector<vector<pair<int, int>>> resultantChunks;
	int chunkCount = noteChunks.size();

	for (int i = 0; i < chunkCount; i++) {
		// Calculate maximum "look forward distance" for chunks to ensure notes stay within the same bar
		int remainder = (i + 1) % 4;
		int distance = 4 - remainder;
		
		if (remainder == 0) {
			distance = 0;
		}

		// Ensure not out of bounds error
		if (i + distance >= chunkCount) {
			distance = chunkCount - 1 - i;
		}

		// Check if notes occur consecutively across the next bar
		vector<pair<int, int>> newChunk;
		int noteCount = currentChunks[i].size();

		for (int j = 0; j < noteCount; j++) {
			pair<int, int> newNote = make_pair(currentChunks[i][j], 1);

			for (int k = 1; k < distance + 1; k++) {
				vector<int> nextChunk = currentChunks[i + k];

				// Check if vector contains note
				vector<int>::const_iterator containsNote = find(nextChunk.begin(), nextChunk.end(), newNote.first);
				if (containsNote != nextChunk.end()) {
					newNote.second += 1;
					
					// Remove note from that future chunk
					int index = containsNote - nextChunk.begin();
					nextChunk.erase(nextChunk.begin() + index);

					currentChunks[i + k] = nextChunk;
				}
				else {
					break; // Notes are not consecutive anymore
				}
			}

			newChunk.push_back(newNote);
		}

		resultantChunks.push_back(newChunk);
	}

	return resultantChunks;
}

void notationViewer::pauseTrack() {
	trackObjectPointer->pause();
	trackPaused = true;
}
void notationViewer::resumeTrack() {
	trackObjectPointer->play();
	trackPaused = false;
}

bool notationViewer::checkIfScroll() {
	int chunkCount = noteLengths.size();
	int chunksPerLine = NOTATION_CHUNKS_PER_LINE * (double(display_x) / 1000.0);
	int requiredStaves = ceilf(float(chunkCount) / float(chunksPerLine));

	float maxHeight = NOTATION_HEIGHT * requiredStaves;
	if (maxHeight > 1.0f) {
		return true;
	}
	return false;
}
mat4 notationViewer::getViewMatrix() {
	mat4 resultantMatrix = mat4(1.0f);

	if (checkIfScroll()) {
		float notationGapDistance = currentLineNumber * NOTATION_HEIGHT * display_y;
		float deltaTime = glfwGetTime() - previousRuntime;

		currentOffset = currentOffset + display_y * deltaTime * NOTATION_SCROLL_RATE;
		if (currentOffset > notationGapDistance) {
			currentOffset = notationGapDistance;
		}

		vec3 translation = vec3(0.0f, currentOffset, 0.0f);
		resultantMatrix = translate(resultantMatrix, translation);
	}

	return resultantMatrix;
}

float notationViewer::getLineLength() {
	float edgeOfTrebleClef = NOTATION_EDGE_DISTANCE + TREBLE_CLEF_WIDTH * aspectRatioMultiplier; // 0.3672 comes from image aspect ratio
	float keySignatureWidth = getKeySignatureWidth(); 

	float lineLength = 1.0f - edgeOfTrebleClef - keySignatureWidth - NOTATION_EDGE_DISTANCE;
	return lineLength;
}
void notationViewer::drawNotes(vector<vector<pair<int, int>>> notes, vector<bool> keySignature) {
	// Prequisite Variables
	int currentStave = 0;
	float currentYPosition = -NOTATION_EDGE_DISTANCE - NOTATION_MAX_LEDGER_LINES * NOTATION_LINE_GAP;

	float edgeOfTrebleClef = NOTATION_EDGE_DISTANCE + TREBLE_CLEF_WIDTH * aspectRatioMultiplier; // 0.3672 comes from image aspect ratio
	float initialNoteXPosition = edgeOfTrebleClef + getKeySignatureWidth();

	int notesPerLine = NOTATION_CHUNKS_PER_LINE * (float(display_x) / 1000.0f); // Notes per line is proportional to screen width
	float noteGapDistance = getLineLength() / float(notesPerLine); // Line Length / notes per line

	// Shift only by natural note distance (because c# for example is no higher on the stave than c)
	vector<int> naturalNoteIndex = { 0, 0, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6 };
	vector<bool> isNaturalNote = { true, false, true, true, false, true, false, true, true, false, true, false };

	// Iterate over note chunks
	int noteCount = notes.size();
	for (int i = 0; i < noteCount; i++) {
		// Calculate x Position of notes on the stave line
		int noteStaveIndex = i % notesPerLine;
		float currentXPosition = initialNoteXPosition + noteGapDistance * noteStaveIndex;
		
		// Iterate over individual notes
		int subNoteCount = notes[i].size();
		for (int j = 0; j < subNoteCount; j++) {
			// Calculate Y Position of Note
			int naturalNote = naturalNoteIndex[notes[i][j].first % 12];
			int naturalNoteBaseNote = naturalNoteIndex[31 % 12]; // 31 is base note E (the current position of Y)

			int octaveDifference = floorf(float(35 - notes[i][j].first) / 12.0f); // 35 is note to the nearest full octave
			int dist = octaveDifference * 7 + (naturalNoteBaseNote - naturalNote);
			float yPosition = currentYPosition - dist * NOTATION_LINE_GAP * 0.5f;
			
			// Check if key signature applies to this note
			bool requiresSharp = false;
			if (isNaturalNote[notes[i][j].first % 12] && !keySignature[naturalNote]) {
				requiresSharp = true;
			}

			 // Render
			float middleOfStave = currentYPosition - NOTATION_LINE_GAP * 2.0F;
			drawSingularNote(vec2(currentXPosition, yPosition), middleOfStave, notes[i][j].second, requiresSharp);
			drawLedgerLines(yPosition, currentYPosition, currentXPosition);			
		}

		// Draw Bar Lines if 4 notes have occured (4 beats per bar)
		if (i % 4 == 0 && i > 0) {
			float barLineXPosition = currentXPosition;
			drawBarLine(barLineXPosition * display_x, currentYPosition * display_y);
		}

		// Check if note is the last of the line, if so calculate new base y position
		if (noteStaveIndex == notesPerLine - 1) {
			currentStave = currentStave + 1;
			currentYPosition = currentYPosition - NOTATION_HEIGHT;
		}
	}
}
void notationViewer::drawNotation() {
	int chunkCount = noteLengths.size();
	int chunksPerLine = NOTATION_CHUNKS_PER_LINE * (double(display_x) / 1000.0);
	int requiredStaves = ceil(double(chunkCount) / chunksPerLine);

	float beatsPerMinute = float(chunkCount) / audioDuration;
	string bpmText = "BPM: " + to_string(beatsPerMinute).substr(0, 4); // Only 2 decimal places
	
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
	drawNotes(noteLengths, keySignature);

	// Draw BPM Text
	vec2 bpmTextPosition = vec2(NOTATION_EDGE_DISTANCE * display_x, display_y - NOTATION_EDGE_DISTANCE * display_y);
	float bpmTextSize = NOTATION_BPM_TEXT_SIZE * (float(display_y) / 1000.0f);

	renderText(bpmText, bpmTextPosition, 1.0f, bpmTextSize, vec3(0.0f), fontCharacters);

	// Update Time Variables
	if (trackPaused) {
		pausedTime = pausedTime + (glfwGetTime() - previousRuntime);
	}
	float playingTime = glfwGetTime() - pausedTime;

	// Calculate Current Line
	float timePerChunk = float(samplesPerChunkProgress) / float(sampleRateProgress);
	float timePerLine = chunksPerLine * timePerChunk;

	int lineNumber = floorf(playingTime / timePerLine);
	currentLineNumber = lineNumber;

	// Calculate progress bar position variables
	drawProgressBar();

	previousRuntime = glfwGetTime();
}

void notationViewer::startProgressBar() {
	progressBarTexture = readyTexture("Assets/progressBar.png");

	float x = PROGRESS_BAR_SIZE * aspectRatioMultiplier * display_x; // 0.22f from image resolution
	float y1 = display_y;
	float y2 = display_y - STAVE_HEIGHT * display_y;

	vector<float> vertices = {
		x, y1, 1.0f, 1.0f,
		x, y2, 1.0f, 0.0f,
		0.0f, y1, 0.0f, 1.0f,

		x, y2, 1.0f, 0.0f,
		0.0f, y2, 0.0f, 0.0f,
		0.0f, y1, 0.0f, 1.0f
	};

	// Start OpenGL Attributes
	glGenVertexArrays(1, &progressBarVAO);
	glGenBuffers(1, &progressBarVBO);

	glBindVertexArray(progressBarVAO);
	glBindBuffer(GL_ARRAY_BUFFER, progressBarVBO);

	// Load Data into Buffer
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
}
void notationViewer::drawProgressBar() {
	// Calculate and limit coordinates
	float minimumX = NOTATION_EDGE_DISTANCE + TREBLE_CLEF_WIDTH * aspectRatioMultiplier + getKeySignatureWidth();
	float lineLength = getLineLength();

	int chunkCount = noteLengths.size();
	float timePerChunk = float(samplesPerChunkProgress) / float(sampleRateProgress);
	int chunksPerLine = NOTATION_CHUNKS_PER_LINE * (double(display_x) / 1000.0);

	float playingTime = glfwGetTime() - pausedTime;
	float timePerLine = chunksPerLine * timePerChunk;

	float lineTime = playingTime - currentLineNumber * timePerLine;
	float lineProportion = lineTime / timePerLine;

	// Calculate limits for when track is finished playing
	int lineCount = ceilf(floor(chunkCount) / float(chunksPerLine));
	float maxYPosition = -NOTATION_EDGE_DISTANCE - NOTATION_MAX_LEDGER_LINES * NOTATION_LINE_GAP - (lineCount - 1) * NOTATION_HEIGHT;

	int finalLineChunkCount = chunkCount % (chunksPerLine + 1);
	float maxXOffset = minimumX + (float(finalLineChunkCount) / float(chunksPerLine)) * lineLength;

	// Calculate final positions and draw
	float xPosition = minimumX + lineProportion * lineLength;
	float yPosition = -NOTATION_EDGE_DISTANCE - NOTATION_MAX_LEDGER_LINES * NOTATION_LINE_GAP - currentLineNumber * NOTATION_HEIGHT;

	// Limits according to how many note chunks
	if (yPosition < maxYPosition) {
		xPosition = maxXOffset;
		yPosition = maxYPosition;
	}
	if (yPosition == maxYPosition) {
		if (xPosition > maxXOffset) {
			xPosition = maxXOffset;
		}
	}

	// Draw
	glUseProgram(imageShader);

	glBindVertexArray(progressBarVAO);
	glBindTexture(GL_TEXTURE_2D, progressBarTexture);

	mat4 scalePositionMatrix = ortho(0.0f, static_cast<GLfloat>(display_x), 0.0f, static_cast<GLfloat>(display_y));
	scalePositionMatrix = translate(scalePositionMatrix, vec3(xPosition * display_x, yPosition * display_y, 0.0f));

	setMat4(imageShader, "scalePositionMatrix", scalePositionMatrix * getViewMatrix());
	glDrawArrays(GL_TRIANGLES, 0, 6);
}