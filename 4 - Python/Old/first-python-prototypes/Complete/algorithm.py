from sys import argv, getsizeof, exit
from requests import post
from youtube_search import YoutubeSearch
from youtube_dl import YoutubeDL
from json import loads
from os import system, remove, rename, rmdir
from base64 import b64encode
from pydub import AudioSegment

pasteApi = "db8bca2c177aa989270f1d2c931ae94f"
pasteUsr = "78fc16a6fc29421cbf97672c49f2affd"

def createPaste(title, content):
    pasteApi = "db8bca2c177aa989270f1d2c931ae94f"
    pasteUsr = "78fc16a6fc29421cbf97672c49f2affd"

    options = {
        "api_dev_key" : pasteApi,
        "api_option" : "paste",
        "api_paste_code" : content,
        "api_user_key" : pasteUsr,
        "api_paste_name" : title,
        "api_paste_private" : "0",
        "api_paste_expire_date" : "10M"
    }

    url = "https://pastebin.com/api/api_post.php"
    result = post(url, data = options)
    print(result.text, getsizeof(content))

def processMP3(songName):
    used = "\"" + songName + "\""

    command = "spleeter separate -i " + used + " -o output"
    system(command)

def downloadMP3(songName, link):
    ydl_opts = {
        'outtmpl' : songName,
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'nocheckcertificate' : True
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])

def searchYoutube(songName):
    result = YoutubeSearch(songName, max_results = 1).to_json()
    resultJSON = loads(result)

    if len(resultJSON["videos"]) == 0:
        print("Error on search!")
        return None

    return "https://www.youtube.com" + resultJSON["videos"][0]["link"]

def convertFiles(directory):
    vocal = AudioSegment.from_wav(directory + "vocals.wav")
    vocal.export(directory + "vocals.mp3", format = "mp3")
    remove(directory + "vocals.wav")

    instrumental = AudioSegment.from_wav(directory + "accompaniment.wav")
    instrumental.export(directory + "instrumental.mp3", format = "mp3")
    remove(directory + "accompaniment.wav")

def encodeAndUpload(directory, name):
    vocalFile = open(directory + "vocals.mp3", "rb")
    vocalEncoded = b64encode(vocalFile.read())

    instrumentalFile = open(directory + "instrumental.mp3", "rb")
    instrumentalEncoded = b64encode(instrumentalFile.read())

    fullFile = open(directory + "full.mp3", "rb")
    fullEncoded = b64encode(fullFile.read())

    createPaste(name + "_vocals", vocalEncoded)
    createPaste(name + "_instrumental", instrumentalEncoded)
    createPaste(name + "_full", fullEncoded)

def removeFiles(directory):
    remove(directory + "vocals.mp3")
    remove(directory + "instrumental.mp3")
    remove(directory + "full.mp3")

    rmdir(directory)

def deletePaste(pasteTitle):
    options = {
        "api_dev_key" : pasteApi,
        "api_user_key" : pasteUsr,
        "api_option" : "list"
    }

    url = "https://pastebin.com/api/api_post.php"
    result = post(url, data = options)

    source = str(result.text)
    if len(source) < 50:
        return

    while source.find("</paste>") != -1:
        start = source.find("<paste_title>") + len("<paste_title>")
        end = source.find("</paste_title>")
        title = source[start : end]

        if title == pasteTitle:
            start = source.find("<paste_url>") + len("<paste_url>")
            end = source.find("</paste_url>")

            pasteUrl = source[start : end]
            start = pasteUrl.rfind("/") + 1
            pasteKey = pasteUrl[start : len(pasteUrl)]

            options = {
                "api_dev_key" : pasteApi,
                "api_user_key" : pasteUsr,
                "api_paste_key" : pasteKey,
                "api_option" : "delete"
            }

            result = post(url, data = options)
            return

        endIndex = source.find("</paste>") + len("</paste>")
        source = source[endIndex : len(source)]

def main():
    songName = argv[1]

    failedAttempts = 0
    link = None

    while failedAttempts < 10:
        link = searchYoutube(songName)
        failedAttempts += 1

    if link == None:
        createPaste(songName + "_error", "body")
        exit(500)
    createPaste(songName + "_foundLink", "body")

    songName = songName + ".mp3"
    downloadMP3(songName, link)

    deletePaste(songName + "_foundLink")
    createPaste(songName + "_downloadedAudio", "body")

    processMP3(songName)

    deletePaste(songName + "_downloadedAudio")
    createPaste(songName + "_processedAudio", "body")

    songName = songName.replace(".mp3", "")
    rename(songName + ".mp3", "output/" + songName + "/" + "full.mp3")

    outputDirectory = "output/" + songName + "/"
    convertFiles(outputDirectory)

    deletePaste(songName + "_processedAudio")
    createPaste(songName + "_convertedAudio", "body")

    encodeAndUpload(outputDirectory, songName)
    deletePaste(songName + "_convertedAudio")
    removeFiles(outputDirectory)

try:
    main()
except():
    createPaste(songName + "_error", "body")
    exit(500)
