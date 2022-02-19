#include "MainProject.h"

int main() {
	srand(time(NULL));

	vector<string> weightDirectories = {
		"vocalsOutputWeights/",
		"bassOutputWeights/",
		"drumsOutputWeights/",
	};
	MainProject newMainProject = MainProject("testTrack.mp3", weightDirectories, 1280, 720);
	
	system("pause");
	return 0;
}