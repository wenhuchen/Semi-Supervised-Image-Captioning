import json

ids = open('result.id')
texts = open('result.text')

results = []
for id, text in zip(ids, texts):
    results.append({'id': int(id.strip()), 'text': text.strip()})

with open("result.json", 'w') as f:
    json.dump(results, f)
