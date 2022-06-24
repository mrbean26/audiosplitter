#include "Headers/splitter.h"

#include <iostream>
using namespace std;

int main() {
	srand(time(NULL));

	splitter newSplitter = splitter(STEM_VOCAL);
	newSplitter.predictTrackStemToFile("test_tracks/riptide.mp3", STEM_VOCAL, "test_tracks/o.wav");

	system("pause");
	return 0;
}