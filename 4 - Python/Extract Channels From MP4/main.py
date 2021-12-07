import glob
allTrainFiles = glob.glob("D:\\Projects\\Audio Splitter App\\Data\\musdb18\\test\\*.mp4")

import os
fileNames = ["mixture.mp3", "drums.mp3", "bass.mp3", "other.mp3", "vocals.mp3"]

for file in allTrainFiles:
    os.mkdir("D:\\Projects\\Audio Splitter App\\Data\\musdb18\\test_channels\\" + file.replace("D:\\Projects\\Audio Splitter App\Data\\musdb18\\test\\", "").replace(".stem.mp4", "") + "\\")

    for i in range(5):
        os.system("cd \"D:\\Projects\\Audio Splitter App\\Data\\musdb18\\test_channels\\" + file.replace("D:\\Projects\\Audio Splitter App\Data\\musdb18\\test\\", "").replace(".stem.mp4", "") + "\\\" && ffmpeg -i " + "\"" + file + "\"" + " -map 0:" + str(i) + " " + fileNames[i])
