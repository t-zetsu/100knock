import requests

url = "http://www.cl.ecei.tohoku.ac.jp/nlp100/data/neko.txt"
output_file = "neko.txt"

response = requests.get(url)
response.encoding = 'UTF-8'

with open(output_file, mode='w', encoding='utf-8') as f:
    f.write(response.text)