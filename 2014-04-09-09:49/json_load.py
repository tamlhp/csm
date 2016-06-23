from __future__ import print_function
import json


#f = open('test.sample.txt')
f = open('test.txt')
#f_json = open('test.sample.json','w')
f_json = open('test.json','w')
for line in f:
    j_content = eval(line)
    dp = json.dumps(j_content)
    print(dp, file=f_json)

f_json.close()