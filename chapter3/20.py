import gzip
import json

input_file = "jawiki-country.json.gz"
output_file = "england.txt"

with gzip.open(input_file, "rt", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if data['title'] == 'イギリス':
            output = data['text']
            break

with open(output_file, mode='w', encoding='utf-8') as f:
    f.writelines(output)