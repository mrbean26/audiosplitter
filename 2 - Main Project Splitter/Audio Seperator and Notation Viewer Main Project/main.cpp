#include "Headers/splitter.h"

#include <iostream>
using namespace std;

int main() {
	srand(time(NULL));
	cout << "BASS" << endl;
	splitter newSplitter = splitter(STEM_BASS);
	//newSplitter.predictTrackStemToFile("test_tracks/riptide.mp3", STEM_VOCAL, "test_tracks/o.wav");
	newSplitter.trainNetwork(STEM_BASS, "_trained_weights/bass/1st_proper_train/");

	system("pause");
	return 0;
}