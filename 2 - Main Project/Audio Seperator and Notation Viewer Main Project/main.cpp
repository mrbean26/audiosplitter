#include "MainProject.h"

int main() {
	srand(time(NULL));

	displayStems({ {} }, "", FAST_QUALITY, 1280, 720);

	system("pause");
	return 0;
}