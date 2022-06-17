#include "Headers/splitter.h"

#include <iostream>
using namespace std;

int main() {
	srand(time(NULL));

	splitter newSplitter = splitter(STEM_VOCAL);
	newSplitter.predictTrackStemToFile("test_tracks/californication.mp3", STEM_VOCAL, "test_tracks/o2.wav");

	system("pause");
	return 0;
}