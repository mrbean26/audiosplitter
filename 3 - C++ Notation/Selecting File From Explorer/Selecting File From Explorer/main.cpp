#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

#include <windows.h>
#include <string.h>
#include <iostream>
#include <filesystem>
#include <experimental/filesystem>

using namespace std;
using namespace experimental;

void openFileName() {
	OPENFILENAME ofn = { 0 };
	TCHAR szFile[MAX_PATH] = { 0 };
	// Initialize remaining fields of OPENFILENAME structure
	ofn.lStructSize = sizeof(ofn);
	//ofn.hwndOwner = hWnd;
	ofn.lpstrFile = szFile;
	ofn.nMaxFile = sizeof(szFile);
	//ofn.nFileExtension = TXT
	ofn.nFilterIndex = 1;
	ofn.lpstrFileTitle = NULL;
	ofn.nMaxFileTitle = 0;
	ofn.lpstrInitialDir = NULL;
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
	//ofn.lpstrFilter = (LPCWSTR)L"Text Files (*.txt)\0*.txt\0All Files (*.*)\0*.*\0";;

	string a = "";
	if (GetOpenFileName(&ofn) == TRUE) {
		filesystem::path myFile = ofn.lpstrFile;
		filesystem::path fullname = myFile.filename();

		for (int i = 0; i < MAX_PATH; i++) {
			a += szFile[i];
		}
	}

	cout << a << endl;
}
void saveFileName() {
	OPENFILENAME ofn;

	char szFileName[MAX_PATH] = { 0 };

	ZeroMemory(&ofn, sizeof(ofn));

	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = NULL;
	//ofn.lpstrFilter = (LPCWSTR)L"Text Files (*.txt)\0*.txt\0All Files (*.*)\0*.*\0";
	ofn.lpstrFile = (LPWSTR)szFileName;
	ofn.nMaxFile = MAX_PATH;
	ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
	ofn.lpstrDefExt = (LPCWSTR)L"txt";

	string a = "";
	if (GetSaveFileName(&ofn) == TRUE) {
		filesystem::path myFile = ofn.lpstrFile;
		filesystem::path fullname = myFile.filename();

		for (int i = 0; i < MAX_PATH; i++) {
			if (szFileName[i] > 0) {
				a += szFileName[i];
			}
			
		}
	}

	cout << a << endl;
}

int main() {
	// Opening Files
	openFileName();

	// Saving Files
	saveFileName();


	system("pause");
	return 0;
}