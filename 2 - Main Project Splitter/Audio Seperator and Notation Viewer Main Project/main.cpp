#include "Headers/splitter.h"

#include <iostream>
using namespace std;

int main() {
	srand(time(NULL));
	
	splitter newSplitter = splitter(STEM_VOCAL);
	newSplitter.predictTrackStemToFile("inputs/1.mp3", STEM_VOCAL, "test_tracks/o.wav");

	system("pause");
	return 0;
}