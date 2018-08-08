import numpy as np
from os.path import isfile, join

# low: 1-3
# high: 7-10

textFile = './labeledSarahahText.txt'
data = open(textFile, 'r').readlines()
low_idx = 0
high_idx = 0
for line in data:
    tmp = line.strip().split('\t')
    text = tmp[0]
    severity = int(tmp[1])
    if (severity==0):
        output = 'tweet/low/'+str(low_idx)+ '.txt'
        out = open(output, 'w+')
        out.write('%s\n' % text)
        low_idx+= 1
        out.close()
    elif (severity==2):
        output = 'tweet/high/'+str(high_idx)+'.txt'
        out = open(output, 'w+')
        out.write('%s\n' % text)
        high_idx += 1
        out.close()

commentFile = './labeledSarahahComment.txt'
data = open(commentFile, 'r').readlines()
low_idx = 0
high_idx = 0
for line in data:
    tmp = line.strip().split('\t')
    text = tmp[0]
    severity = int(tmp[1])
    if (severity==0):
        output = 'comment/low/'+str(low_idx)+ '.txt'
        out = open(output, 'w+')
        out.write('%s\n' % text)
        low_idx+= 1
        out.close()
    elif (severity==2):
        output = 'comment/high/'+str(high_idx)+'.txt'
        out = open(output, 'w+')
        out.write('%s\n' % text)
        high_idx += 1
        out.close()

print low_idx