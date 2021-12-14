#include <iostream>
#include <vector>
using namespace std;

vector<vector<int>> returnedFrets(vector<vector<int>> notes, vector<int> tunings, vector<int> maxFrets) {
	vector<vector<int>> result;
	int stringCount = tunings.size();
	int frameCount = notes.size();

	for (int i = 0; i < frameCount; i++) {
		vector<int> currentFrame(stringCount);
		std::fill(currentFrame.begin(), currentFrame.end(), -1); // Fill with -1's, meaning no fret

		// Loop over all notes in frame
		int noteCount = notes[i].size();
		for (int j = 0; j < noteCount; j++) {
			int lowestDifference = INT_MAX;
			int chosenStringIndex = -1;

			for (int k = 0; k < stringCount; k++) {
				int currentDifference = notes[i][j] - tunings[k]; // Find Fret Number of Note on this String

				// Find lowest fret number
				if (currentDifference < lowestDifference && currentDifference >= 0) {
					if (currentDifference <= tunings[k]) {
						chosenStringIndex = k;
						lowestDifference = currentDifference;
					}
				}
			}

			// Assign note to a fret on chosen string
			for (int k = chosenStringIndex; k >= 0; k--) {
				if (currentFrame[k] == -1) {
					currentFrame[k] = notes[i][j] - tunings[k];
					break;
				}

				// If the string is already occupied by a note, push note up with lowest fret
				int newNoteFret = notes[i][j] - tunings[k];
				int currentNoteFret = currentFrame[k];

				if (currentNoteFret < newNoteFret) {
					notes[i][j] = currentNoteFret;
				}
			}
		}
		result.push_back(currentFrame);
	}
	return result;
}

int main() {
	// Config
	vector<int> tunings = { 7, 12, 17, 22, 26, 31 }; // Guitar Standard Tuning
	vector<int> maxFrets = { 21, 21, 21, 21, 21, 21 };
	
	// Note Frames
	vector<vector<int>> noteFrames = { {15, 19, 22, 27, 31} }; // Guitar C Chord (C, E, G, C, E)
	vector<vector<int>> finalFrets = returnedFrets(noteFrames, tunings, maxFrets);

	// Output
	for (int i = 0; i < 6; i++) {
		cout << finalFrets[0][i] << endl;
	}
	system("pause");
}