#include "Headers/splitter.h"

#include <iostream>
using namespace std;

int main() {
	srand(time(NULL));
	
	splitter newSplitter = splitter(STEM_VOCAL);
	
	newSplitter.splitStems(STEMS_VOCALS_BACKING, "rhcpSongs/Californication.mp3", "rhcpSongs/Californication/");
	newSplitter.splitStems(STEMS_VOCALS_BACKING, "rhcpSongs/Snow.mp3", "rhcpSongs/Snow/");
	newSplitter.splitStems(STEMS_VOCALS_BACKING, "rhcpSongs/Dani California.mp3", "rhcpSongs/Dani California/");
	newSplitter.splitStems(STEMS_VOCALS_BACKING, "rhcpSongs/Wet Sand.mp3", "rhcpSongs/Wet Sand/");
	newSplitter.splitStems(STEMS_VOCALS_BACKING, "rhcpSongs/Under The Bridge.mp3", "rhcpSongs/Under The Bridge/");
	newSplitter.splitStems(STEMS_VOCALS_BACKING, "rhcpSongs/Scar Tissue.mp3", "rhcpSongs/Scar Tissue/");

	system("pause");
	return 0;
}