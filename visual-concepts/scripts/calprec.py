from __future__ import division
import json
import sys


ids = map(lambda x: int(x.strip()), open('result.id').readlines())
texts = map(lambda x: x.split(), open('result.text').readlines())
test_caption = json.load(open(sys.argv[1]))
test_caption_hash = {}

for item in test_caption: 
    test_caption_hash[item['id']] = item['words']

prec = 0
for id, text in zip(ids, texts):
    ref = test_caption_hash[id]
    #import pdb
    #pdb.set_trace()
    prec += len(set(ref) & set(text))/len(ref)

print "precision: %0.2f"%(prec/5000,)
