#include "Headers/splitter.h"

#include <iostream>
using namespace std;

int main() {
	srand(time(NULL));
	
	splitter newSplitter = splitter(STEM_VOCAL);
	newSplitter.splitStems(STEMS_ALL, "test_tracks/riptide.mp3", "test_vocal_backing/");

	system("pause");
	return 0;
}