from subprocess import Popen
from socket import create_connection

from requests import post
pasteApi = "db8bca2c177aa989270f1d2c931ae94f"
pasteUsr = "78fc16a6fc29421cbf97672c49f2affd"

def hasInternet():
    try:
        create_connection(("www.google.com", 80))
        return True
    except:
        return False

def getPasteRequest():
    options = {
        "api_dev_key" : pasteApi,
        "api_user_key" : pasteUsr,
        "api_option" : "list"
    }

    url = "https://pastebin.com/api/api_post.php"
    result = post(url, data = options)

    source = str(result.text)
    if len(source) < 50:
        return None, None

    while source.find("</paste>") != -1:
        start = source.find("<paste_title>") + len("<paste_title>")
        end = source.find("</paste_title>")
        title = source[start : end]

        if title == "New Request":
            start = source.find("<paste_url>") + len("<paste_url>")
            end = source.find("</paste_url>")

            pasteUrl = source[start : end]
            start = pasteUrl.rfind("/") + 1
            pasteKey = pasteUrl[start : len(pasteUrl)]

            url = "https://pastebin.com/api/api_raw.php"
            options = {
                "api_dev_key" : pasteApi,
                "api_user_key" : pasteUsr,
                "api_paste_key" : pasteKey,
                "api_option" : "show_paste"
            }

            result = post(url, data = options)
            return result.text, pasteKey

        endIndex = source.find("</paste>") + len("</paste>")
        source = source[endIndex : len(source)]

    return None, None

def deletePaste(pasteKey):
    url = "https://pastebin.com/api/api_post.php"
    options = {
        "api_dev_key" : pasteApi,
        "api_user_key" : pasteUsr,
        "api_paste_key" : pasteKey,
        "api_option" : "delete"
    }

    result = post(url, data = options)

while True:
    if hasInternet() == True:
        newPasteEntry, newPasteKey = getPasteRequest()
        if newPasteEntry == None:
            continue

        print(newPasteEntry)
        deletePaste(newPasteKey)
        newCommand = ["python", "algorithm.py", newPasteEntry]
        Popen(newCommand)
    else:
        print("No internet.")
