import re
import requests
input_file = "england.txt"

def find_key(line, key):
    flag = 0
    for k in key:
        if k in line:
            flag = 1
    return flag

def startswith_key(line, key):
    flag = 0
    for k in key:
        if line.startswith(k):
            flag = 1
    return flag

def get_idx(line, list_s):
    for s in list_s:
        if s in line:
            idx = line.find(s)+len(s)
    return idx

def remove_characters(line, character_list):
    for c in character_list:
        if c in line:
            line = line.replace(c,"")
    return line

def get_image_url(filename):
    url = "https://www.mediawiki.org/w/api.php"
    params = {
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "iiprop": "url",
    "titles": "File:" + filename
    }
    result = requests.get(url, params).json()
    return result['query']['pages']['-1']['imageinfo'][0]['url']

def main():
    bracket_flag = 0
    basic_information = {}
    character_list = ["'","|","[[","]]","\n"]
    with open(input_file, mode='r', encoding='utf-8') as f:
        for line in f:
            if startswith_key(line,["{{"]) * find_key(line, ["基礎情報"]) == 1:
                bracket_flag = 1
            if startswith_key(line,["}}"]) == 1:
                bracket_flag = 0
            if bracket_flag * startswith_key(line, ["|"]) == 1:
                line = remove_characters(line,character_list)
                line = re.sub("\[.+?\]","",line)
                line = re.sub("\<.+?\>","",line)
                split_idx = get_idx(line, ["="])
                key = line[:split_idx - 1].replace(" ","")
                value = line[split_idx:]
                basic_information[key] = value
        url = get_image_url(basic_information["国旗画像"])
        print(url)

main()