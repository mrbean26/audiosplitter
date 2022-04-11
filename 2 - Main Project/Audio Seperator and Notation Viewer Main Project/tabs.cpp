#include "Headers/tabs.h"
#include "Headers/audio.h"
#include "Headers/graphics.h"

tabViewer::tabViewer() {

}
tabViewer::tabViewer(vector<vector<vector<int>>> notes, vector<int> tunings, vector<int> maxFrets, int stringCount, int samplesPerChunk, int sampleRate, vector<audioObject*> trackAudio) {
	int stemCount = notes.size();
	for (int i = 0; i < stemCount; i++) {
		noteFrets.push_back(notesToFrets(notes[i], tunings, maxFrets));
	}

	tabStringCount = stringCount;
	tabsBegin(stringCount);

	samplesPerChunkProgress = samplesPerChunk;
	sampleRateProgress = sampleRate;

	trackObjectPointer = trackAudio;
	tabHeight = (stringCount - 1) * TAB_LINE_GAP;

	originalNotes = notes;
}

void tabViewer::changeInstrumentConfig(vector<int> tunings, vector<int> maxFrets) {
	// Calculate new note frets
	int stemCount = originalNotes.size();
	noteFrets.clear();

	for (int i = 0; i < stemCount; i++) {
		noteFrets.push_back(notesToFrets(originalNotes[i], tunings, maxFrets));
	}

	tabStringCount = tunings.size();
	tabHeight = (tabStringCount - 1) * TAB_LINE_GAP;

	// Generate new string lines vertices
	vector<float> vertices = {};
	for (int j = 0; j < tabStringCount; j++) {
		vertices.push_back(display_x * TAB_EDGE_DISTANCE); // Start of Line
		vertices.push_back(display_y - (j * TAB_LINE_GAP * display_y));

		vertices.push_back(display_x - display_x * TAB_EDGE_DISTANCE);
		vertices.push_back(display_y - (j * TAB_LINE_GAP * display_y));
	}

	glBindVertexArray(tabVAOs[0]);
	glBindBuffer(GL_ARRAY_BUFFER, tabVBOs[0]);

	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	tabSizes[0] = vertices.size() / 2;

	// Generate new progress bar
	float x = PROGRESS_BAR_SIZE * aspectRatioMultiplier * tabHeight * display_x; // 0.22f from image resolution
	float y1 = display_y;
	float y2 = display_y - tabHeight * display_y;

	vertices = {
		x, y1, 1.0f, 1.0f,
		x, y2, 1.0f, 0.0f,
		0.0f, y1, 0.0f, 1.0f,

		x, y2, 1.0f, 0.0f,
		0.0f, y2, 0.0f, 0.0f,
		0.0f, y1, 0.0f, 1.0f
	};

	glBindVertexArray(progressBarVAO);
	glBindBuffer(GL_ARRAY_BUFFER, progressBarVBO);

	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
}

void tabViewer::tabsBegin(int stringCount) {
	// Begin Shader
	int vertShader = createShader("Assets/Shaders/tabVert.txt", GL_VERTEX_SHADER);
	int fragShader = createShader("Assets/Shaders/tabFrag.txt", GL_FRAGMENT_SHADER);
	tabShader = createProgram({ vertShader, fragShader });
	
	// Image Shader
	vertShader = createShader("Assets/Shaders/textureVert.txt", GL_VERTEX_SHADER);
	fragShader = createShader("Assets/Shaders/textureFrag.txt", GL_FRAGMENT_SHADER);
	imageShader = createProgram({ vertShader, fragShader });

	// Begin Coordinates
	//int minStringCount = stringCounts[0];

	tabVAOs.push_back(0); tabVBOs.push_back(0);
	vector<float> vertices = {};

	for (int j = 0; j < stringCount; j++) {
		vertices.push_back(display_x * TAB_EDGE_DISTANCE); // Start of Line
		vertices.push_back(display_y - (j * TAB_LINE_GAP * display_y));

		vertices.push_back(display_x - display_x * TAB_EDGE_DISTANCE);
		vertices.push_back(display_y - (j * TAB_LINE_GAP * display_y));
	}

	// Ready OpenGL Attributes
	GLuint tabSize = readyVertices(&tabVAOs[0], &tabVBOs[0], vertices, 2);
	tabSizes.push_back(tabSize);

	// Begin progress bar
	progressBarTexture = readyTexture("Assets/progressBar.png");
	tabHeight = (stringCount - 1) * TAB_LINE_GAP;

	float x = PROGRESS_BAR_SIZE * aspectRatioMultiplier * tabHeight * display_x; // 0.22f from image resolution
	float y1 = display_y;
	float y2 = display_y - tabHeight * display_y;

	vertices = {
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
void tabViewer::drawTabLines(int index, float yOffset) {
	glUseProgram(tabShader);

	mat4 projectionMatrix = ortho(0.0f, static_cast<GLfloat>(display_x), 0.0f, static_cast<GLfloat>(display_y));
	projectionMatrix = translate(projectionMatrix, vec3(0.0f, yOffset, 0.0f));
	setMat4(tabShader, "projection", projectionMatrix * getViewMatrix());

	glBindVertexArray(tabVAOs[index]);
	glDrawArrays(GL_LINES, 0, tabSizes[index]);
}

void tabViewer::pauseTrack() {
	trackObjectPointer[currentStem]->pause();
	trackPaused = true;
}
void tabViewer::resumeTrack() {
	trackObjectPointer[currentStem]->play();
	trackPaused = false;
}

bool tabViewer::checkIfScroll() {
	// Check if tab viewer is more than screen
	float chunkHeight = tabHeight + TAB_EDGE_DISTANCE;
	int chunksPerLine = TAB_CHUNKS_PER_LINE * (float(display_x) / 1000.0f);

	int chunkCount = noteFrets[currentStem].size();
	int requiredLines = ceilf(float(chunkCount) / float(chunksPerLine));

	float maxHeight = chunkHeight * requiredLines;
	if (maxHeight > 1.0f) {
		// Longer than screen
		return true;
	}
	return false;
}
mat4 tabViewer::getViewMatrix() {
	mat4 resultantMatrix = mat4(1.0f);
	
	if (checkIfScroll()) {
		float tapGapDistance = currentLineNumber * (tabHeight + TAB_EDGE_DISTANCE) * display_y;
		float deltaTime = glfwGetTime() - previousRuntime;

		currentOffset = currentOffset + display_y * deltaTime * TAB_SCROLL_RATE;
		if (currentOffset > tapGapDistance) {
			currentOffset = tapGapDistance;
		}

		vec3 translation = vec3(0.0f, currentOffset, 0.0f);
		resultantMatrix = translate(resultantMatrix, translation);
	}

	return resultantMatrix;
}
void tabViewer::drawTab() {
	int chunkCount = noteFrets[currentStem].size();
	int chunksPerLine = TAB_CHUNKS_PER_LINE * (float(display_x) / 1000.0f);

	float tabTextSize = TAB_TEXT_SIZE * (float(display_y) / 1000.0f);
	float tabTextDistance = LINE_LENGTH / float(chunksPerLine);
	
	// Draw Lines
	int tabLinesCount = ceil(float(chunkCount) / float(chunksPerLine));
	for (int i = 0; i < tabLinesCount; i++) {
		float yCoordinate = -TAB_EDGE_DISTANCE;
		yCoordinate = yCoordinate - i * (tabStringCount * TAB_LINE_GAP + TAB_EDGE_DISTANCE);

		drawTabLines(0, yCoordinate * display_y);
	}

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

	drawProgressBar();

	// Draw Text 
	int characterCount = 0;
	int stringCount = noteFrets[currentStem][0].size();

	for (int i = 0; i < chunkCount; i++) {
		int lineIndex = i % chunksPerLine;
		int tabChunkIndex = floor(float(i) / float(chunksPerLine));

		float relativeXPosition = TAB_EDGE_DISTANCE + lineIndex * tabTextDistance;
		
		for (int j = 0; j < stringCount; j++) {
			if (noteFrets[currentStem][i][j] == -1) {
				continue;
			}

			float relativeYPosition = 1.0f -TAB_EDGE_DISTANCE - TAB_EDGE_DISTANCE * tabChunkIndex - stringCount * TAB_LINE_GAP * tabChunkIndex;
			relativeYPosition = relativeYPosition - TAB_LINE_GAP * (stringCount - j - 1);

			vec2 position = vec2(relativeXPosition * display_x, relativeYPosition * display_y);
			if (foundSize) {
				position = position - vec2(0.0f, averageYCharacterSize / 2.0f); // Center Text
			}
			// shift text for scrolling
			if (checkIfScroll()) {
				position.y += currentOffset;
			}

			vec2 textWidthHeight = renderText(to_string(noteFrets[currentStem][i][j]), position, 1.0f, tabTextSize, vec3(0.0f), fontCharacters);
			if (!foundSize) {
				averageYCharacterSize = averageYCharacterSize + textWidthHeight.y;
				characterCount++;
			}
		}
	}
	if (!foundSize) {
		averageYCharacterSize = averageYCharacterSize / float(characterCount);
		foundSize = true;
	}
}

void tabViewer::drawProgressBar() {
	// Calculate progress bar position variables
	float timePerChunk = float(samplesPerChunkProgress) / float(sampleRateProgress);
	int chunksPerLine = TAB_CHUNKS_PER_LINE * (double(display_x) / 1000.0);
	float timePerLine = chunksPerLine * timePerChunk;
	int chunkCount = noteFrets[currentStem].size();

	float playingTime = glfwGetTime() - pausedTime;

	int lineNumber = floorf(playingTime / timePerLine);
	currentLineNumber = lineNumber;
	float usedTime = playingTime - currentLineNumber * timePerLine;

	float lineProportion = usedTime / timePerLine;
	float xOffset = lineProportion * LINE_LENGTH;

	// Calculate limits for when track is finished playing
	int lineCount = ceilf(floor(chunkCount) / float(chunksPerLine));
	float maxYPosition = -TAB_EDGE_DISTANCE - (lineCount - 1) * (tabStringCount * TAB_LINE_GAP + TAB_EDGE_DISTANCE);

	int finalLineChunkCount = chunkCount % (chunksPerLine + 1);
	float maxXOffset = (float(finalLineChunkCount) / float(chunksPerLine)) * LINE_LENGTH;

	// Calculate final positions and draw
	float xPosition = TAB_EDGE_DISTANCE + xOffset;
	float yPosition = -TAB_EDGE_DISTANCE - currentLineNumber * (tabStringCount * TAB_LINE_GAP + TAB_EDGE_DISTANCE);

	if (yPosition < maxYPosition) {
		xPosition = TAB_EDGE_DISTANCE + maxXOffset;
		yPosition = maxYPosition;
	}
	if (yPosition == maxYPosition) {
		if (xOffset > maxXOffset) {
			xPosition = TAB_EDGE_DISTANCE + maxXOffset;
		}
	}

	glUseProgram(imageShader);

	glBindVertexArray(progressBarVAO);
	glBindTexture(GL_TEXTURE_2D, progressBarTexture);

	mat4 scalePositionMatrix = ortho(0.0f, static_cast<GLfloat>(display_x), 0.0f, static_cast<GLfloat>(display_y));
	scalePositionMatrix = translate(scalePositionMatrix, vec3(xPosition * display_x, yPosition * display_y, 0.0f));

	setMat4(imageShader, "scalePositionMatrix", scalePositionMatrix * getViewMatrix());
	glDrawArrays(GL_TRIANGLES, 0, 6);
}