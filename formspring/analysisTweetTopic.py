import csv
import numpy as np
from os.path import isfile, join
from six.moves import cPickle as pickle

import string
import pandas as pd

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import os
import datetime
import time

import enchant


# tweet_file = '/home/yue/Public/C_program/cl2_project/SeededLDA/data/' \
#              'sarahah_data/new_result3/topics/topic0.txt'
# tweet_file = './dataset/stem_data.txt'
# bestTopic_file = '/home/yue/Public/C_program/cl2_project/' \
#                  'SeededLDA/data/sarahah_data/SeededLDA_bestTopic.txt'

tweet_file = './formspring_data.txt'
# tweet_file = './userComments/comments0.txt'
bestTopic_file = '/home/yue/Public/C_program/cl2_project/' \
                 'SeededLDA/data/formspring_data/SeededLDA_bestTopic.txt'

# output = './tmp_output.txt'
# out = open(output, 'w+')

output0 = './topics/topic0.txt'
out0 = open(output0, 'w+')

output1 = './topics/topic1.txt'
out1 = open(output1, 'w+')

output2 = './topics/topic2.txt'
out2 = open(output2, 'w+')

# output3 = './topics/topic3.txt'
# out3 = open(output3, 'w+')
#
# output4 = './topics/topic4.txt'
# out4 = open(output4, 'w+')
#
# output5 = './topics/topic5.txt'
# out5 = open(output5, 'w+')

data1 = open(tweet_file, 'r').readlines()
data2 = open(bestTopic_file, 'r').readlines()

for i in range(len(data1)):
    if i ==0:
        continue
    line1 = data1[i].strip().split('\t')
    rec = line1[3]
    id = line1[0]
    line2 = data2[i-1].strip()

    if line2 == '0':
        out0.write('%s\t%s\t%s\n' % (id, rec, line2))
    elif line2 == '1':
        out1.write('%s\t%s\t%s\n' % (id, rec, line2))
    elif line2 == '2':
        out2.write('%s\t%s\t%s\n' % (id, rec, line2))
    # elif line2 == '3':
    #     out3.write('%s\t%s\t%s\n' % (line1, line2))
    # elif line2 == '4':
    #     out4.write('%s\t%s\n' % (line1, line2))
    # elif line2 == '5':
    #     out5.write('%s\t%s\n' % (line1, line2))

    # record = line1.split('\t')
    # seq_num = int(record[0])
    # tweet = record[1]
    #
    # # if line2 == '0':
    # out.write('%s\t%s\n' % (tweet, line2))
