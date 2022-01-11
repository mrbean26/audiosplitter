#include "Headers/tabs.h"
#include "Headers/audio.h"
#include "Headers/graphics.h"

tabViewer::tabViewer(vector<vector<int>> notes, vector<int> tunings, vector<int> maxFrets, vector<int> stringCounts) {
	noteFrets = notesToFrets(notes, tunings, maxFrets);
	tabsBegin(stringCounts);
}

void tabViewer::tabsBegin(vector<int> stringCounts) {
	// Begin Shader
	int vertShader = createShader("Assets/Shaders/tabVert.txt", GL_VERTEX_SHADER);
	int fragShader = createShader("Assets/Shaders/tabFrag.txt", GL_FRAGMENT_SHADER);
	tabShader = createProgram({ vertShader, fragShader });

	// Begin Coordinates
	int stringTypes = stringCounts.size();

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
	}
}
void tabViewer::drawTabLines(int index, float yOffset) {
	glUseProgram(tabShader);

	mat4 projectionMatrix = ortho(0.0f, static_cast<GLfloat>(display_x), 0.0f, static_cast<GLfloat>(display_y));
	projectionMatrix = translate(projectionMatrix, vec3(0.0f, yOffset, 0.0f));
	setMat4(tabShader, "projection", projectionMatrix);

	glBindVertexArray(tabVAOs[index]);
	glDrawArrays(GL_LINES, 0, tabSizes[index]);
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

	// Draw Text
	int characterCount = 0;
	for (int i = 0; i < chunkCount; i++) {
		int lineIndex = i % (chunksPerLine + 1);
		int tabChunkIndex = floor(float(i) / float(chunksPerLine + 1));

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