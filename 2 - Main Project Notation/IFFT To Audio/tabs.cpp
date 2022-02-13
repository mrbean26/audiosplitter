#include "Headers/tabs.h"
#include "Headers/audio.h"
#include "Headers/graphics.h"

tabViewer::tabViewer(vector<vector<int>> notes, vector<int> tunings, vector<int> maxFrets, vector<int> stringCounts, int samplesPerChunk, int sampleRate, audioObject* trackAudio) {
	noteFrets = notesToFrets(notes, tunings, maxFrets);
	tabsBegin(stringCounts);

	samplesPerChunkProgress = samplesPerChunk;
	sampleRateProgress = sampleRate;

	trackObjectPointer = trackAudio;
}

void tabViewer::tabsBegin(vector<int> stringCounts) {
	// Begin Shader
	int vertShader = createShader("Assets/Shaders/tabVert.txt", GL_VERTEX_SHADER);
	int fragShader = createShader("Assets/Shaders/tabFrag.txt", GL_FRAGMENT_SHADER);
	tabShader = createProgram({ vertShader, fragShader });

	// Image Shader
	vertShader = createShader("Assets/Shaders/textureVert.txt", GL_VERTEX_SHADER);
	fragShader = createShader("Assets/Shaders/textureFrag.txt", GL_FRAGMENT_SHADER);
	imageShader = createProgram({ vertShader, fragShader });

	// Begin Coordinates
	int stringTypes = stringCounts.size();
	int minStringCount = stringCounts[0];

	for (int i = 0; i < stringTypes; i++) {
		tabVAOs.push_back(0); tabVBOs.push_back(0);
		vector<float> vertices = {};

		for (int j = 0; j < stringCounts[i]; j++) {
			vertices.push_back(display_x * TAB_EDGE_DISTANCE); // Start of Line
			vertices.push_back(display_y - (j * TAB_LINE_GAP * display_y));

			vertices.push_back(display_x - display_x * TAB_EDGE_DISTANCE);
			vertices.push_back(display_y - (j * TAB_LINE_GAP * display_y));
		}

		// Ready OpenGL Attributes
		GLuint tabSize = readyVertices(&tabVAOs[i], &tabVBOs[i], vertices, 2);
		tabSizes.push_back(tabSize);

		minStringCount = std::min(stringCounts[i], minStringCount); // For progress bar size
	}

	// Begin progress bar
	progressBarTexture = readyTexture("Assets/progressBar.png");

	float tabHeight = (minStringCount - 1) * TAB_LINE_GAP;
	float aspectRatioMultiplier = (float(display_y) / float(display_x));

	float x = 0.22f * aspectRatioMultiplier * tabHeight * display_x; // 0.22f from image resolution
	float y1 = display_y;
	float y2 = display_y - tabHeight * display_y;

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
void tabViewer::drawTabLines(int index, float yOffset) {
	glUseProgram(tabShader);

	mat4 projectionMatrix = ortho(0.0f, static_cast<GLfloat>(display_x), 0.0f, static_cast<GLfloat>(display_y));
	projectionMatrix = translate(projectionMatrix, vec3(0.0f, yOffset, 0.0f));
	setMat4(tabShader, "projection", projectionMatrix * getViewMatrix());

	glBindVertexArray(tabVAOs[index]);
	glDrawArrays(GL_LINES, 0, tabSizes[index]);
}

void tabViewer::pauseTrack() {
	trackObjectPointer->pause();
	trackPaused = true;
}
void tabViewer::resumeTrack() {
	trackObjectPointer->play();
	trackPaused = false;
}

mat4 tabViewer::getViewMatrix() {
	mat4 resultantMatrix = mat4(1.0f);
	
	float tapGapDistance = currentLineNumber * (6 * TAB_LINE_GAP + TAB_EDGE_DISTANCE) * display_y;
	vec3 translation = vec3(0.0f, tapGapDistance, 0.0f);

	resultantMatrix = translate(resultantMatrix, translation);
	return resultantMatrix;
}
void tabViewer::drawTab() {
	int chunkCount = noteFrets.size();
	int stringCount = noteFrets[0].size();

	float lineLength = 1.0f - 2 * TAB_EDGE_DISTANCE;
	int chunksPerLine = TAB_CHUNKS_PER_LINE * (float(display_x) / 1000.0f);

	float tabTextSize = TAB_TEXT_SIZE * (float(display_y) / 1000.0f);
	float tabTextDistance = lineLength / float(chunksPerLine);
	
	// Draw Lines
	int tabLinesCount = ceil(float(chunkCount) / float(chunksPerLine));
	for (int i = 0; i < tabLinesCount; i++) {
		float yCoordinate = -TAB_EDGE_DISTANCE;
		yCoordinate = yCoordinate - i * (6 * TAB_LINE_GAP + TAB_EDGE_DISTANCE);

		drawTabLines(0, yCoordinate * display_y);
	}

	// Pausing and playing
	float deltaTime = glfwGetTime() - previousRuntime;
	previousRuntime = glfwGetTime();

	if (trackPaused) {
		pausedTime = pausedTime + deltaTime;
	}

	// Calculate progress bar position variables
	float timePerChunk = float(samplesPerChunkProgress) / float(sampleRateProgress);
	float timePerLine = chunksPerLine * timePerChunk;

	float playingTime = glfwGetTime() - pausedTime;

	int lineNumber = floorf(playingTime / timePerLine);
	currentLineNumber = lineNumber;
	float usedTime = playingTime - currentLineNumber * timePerLine;

	float lineProportion = usedTime / timePerLine;
	float xOffset = lineProportion * lineLength;

	// Calculate limits for when track is finished playing
	int lineCount = ceilf(floor(chunkCount) / float(chunksPerLine));
	float maxYPosition = -TAB_EDGE_DISTANCE - (lineCount - 1) * (6 * TAB_LINE_GAP + TAB_EDGE_DISTANCE);

	int finalLineChunkCount = chunkCount % (chunksPerLine + 1);
	float maxXOffset = (float(finalLineChunkCount) / float(chunksPerLine)) * lineLength;

	// Calculate final positions and draw
	float xPosition = TAB_EDGE_DISTANCE + xOffset;
	float yPosition = -TAB_EDGE_DISTANCE - currentLineNumber * (6 * TAB_LINE_GAP + TAB_EDGE_DISTANCE);

	if (yPosition <= maxYPosition) {
		if (xOffset >= maxXOffset) {
			xPosition = TAB_EDGE_DISTANCE + maxXOffset;
			yPosition = maxYPosition;
		}
	}

	drawProgressBar(xPosition * display_x, yPosition * display_y);

	// Draw Text 
	int characterCount = 0;
	for (int i = 0; i < chunkCount; i++) {
		int lineIndex = i % (chunksPerLine + 1);
		int tabChunkIndex = floor(float(i) / float(chunksPerLine));

		float relativeXPosition = TAB_EDGE_DISTANCE + lineIndex * tabTextDistance;

		for (int j = 0; j < stringCount; j++) {
			if (noteFrets[i][j] == -1) {
				continue;
			}

			float relativeYPosition = 1.0f - TAB_EDGE_DISTANCE - TAB_EDGE_DISTANCE * tabChunkIndex - stringCount * TAB_LINE_GAP * tabChunkIndex;
			relativeYPosition = relativeYPosition - TAB_LINE_GAP * (stringCount - j);

			vec2 position = vec2(relativeXPosition * display_x, relativeYPosition * display_y);
			if (foundSize) {
				position = position + vec2(0.0f, averageYCharacterSize / 2.0f); // Center Text
			}
			// shift text for scrolling
			position.y += currentLineNumber * (6 * TAB_LINE_GAP + TAB_EDGE_DISTANCE) * display_y;

			vec2 textWidthHeight = renderText(to_string(noteFrets[i][j]), position, 1.0f, tabTextSize, vec3(0.0f), fontCharacters);
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

void tabViewer::drawProgressBar(float xOffset, float yOffset) {
	glUseProgram(imageShader);

	glBindVertexArray(progressBarVAO);
	glBindTexture(GL_TEXTURE_2D, progressBarTexture);

	mat4 scalePositionMatrix = ortho(0.0f, static_cast<GLfloat>(display_x), 0.0f, static_cast<GLfloat>(display_y));
	scalePositionMatrix = translate(scalePositionMatrix, vec3(xOffset, yOffset, 0.0f));

	setMat4(imageShader, "scalePositionMatrix", scalePositionMatrix * getViewMatrix());
	glDrawArrays(GL_TRIANGLES, 0, 6);
}