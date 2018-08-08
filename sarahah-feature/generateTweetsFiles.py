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

stem_dict = {}
stem_file = './dataset/src_data.txt'
stem_data = open(stem_file, 'r').readlines()
for line in stem_data:
    record = line.strip().split('\t')
    seq_num = int(record[0])
    tweet = record[1]

    stem_dict[seq_num] = tweet


topic0_file = '/home/yue/Public/C_program/cl2_project/SeededLDA/data/sarahah_data' \
              '/new_result3/topics/topic0.txt'
#'topics/explicit_se*_topic.txt'
topic1_file = '/home/yue/Public/C_program/cl2_project/SeededLDA/data/sarahah_data' \
              '/new_result3/topics/topic1.txt'
#'topics/haterAndAbuse_topic.txt'
topic2_file = '/home/yue/Public/C_program/cl2_project/SeededLDA/data/sarahah_data' \
              '/new_result3/subtopic2/topics/topic2.txt'
#'topics/flirting_topic.txt'
topic3_file = '/home/yue/Public/C_program/cl2_project/SeededLDA/data/sarahah_data' \
              '/new_result3/subtopic2/topics/topic3.txt'

nonBully_file = '/home/yue/Public/C_program/cl2_project/SeededLDA/data/sarahah_data' \
                 '/new_result3/subtopic2/topics/rest_topic.txt'


topic0_seqList = []
topic1_seqList = []

# output0 = './userComments/comments0.txt'
# out0 = open(output0, 'w+')
topic0_data = open(topic0_file, 'r').readlines()
idx = 0
for line in topic0_data:
    record = line.strip().split('\t')
    seq_num = int(record[0])
    tweet = stem_dict[seq_num]
    if len(tweet) != 0:
        out0 = open('./topics/topic0_tweets/'+ str(idx)+'.txt', 'w+')
        out0.write('%s\n' % tweet)
        out0.close()
        idx += 1

# output1 = './userComments/comments1.txt'
# out1 = open(output1, 'w+')
topic1_data = open(topic1_file, 'r').readlines()
idx = 0
for line in topic1_data:
    record = line.strip().split('\t')
    seq_num = int(record[0])
    tweet = stem_dict[seq_num]
    if len(tweet) != 0:
        out1 = open('./topics/topic1_tweets/' + str(idx)+'.txt', 'w+')
        out1.write('%s\n' % tweet)
        out1.close()
        idx += 1

# output2 = './userComments/comments2.txt'
# out2 = open(output2, 'w+')
topic2_data = open(topic2_file, 'r').readlines()
idx = 0
for line in topic2_data:
    record = line.strip().split('\t')
    seq_num = int(record[0])
    tweet = stem_dict[seq_num]

    if len(tweet) != 0:
        out2 = open('./topics/topic2_tweets/'+str(idx)+'.txt', 'w+')
        out2.write('%s\n' % tweet)
        out2.close()
        idx += 1

# output3 = './userComments/comments3.txt'
# out3 = open(output3, 'w+')
topic3_data = open(topic3_file, 'r').readlines()
idx = 0
for line in topic3_data:
    record = line.strip().split('\t')
    seq_num = int(record[0])
    tweet = stem_dict[seq_num]
    if len(tweet) != 0:
        out3 = open('./topics/topic3_tweets/'+str(idx)+'.txt', 'w+')
        out3.write('%s\n' % tweet)
        out3.close()
        idx += 1


nonBully_data = open(nonBully_file, 'r').readlines()
idx = 0
for line in nonBully_data:
    record = line.strip().split('\t')
    seq_num = int(record[0])
    tweet = stem_dict[seq_num]

    if len(tweet) != 0:
        out4 = open('./topics/topic4_tweets/'+str(idx)+'.txt', 'w+')
        out4.write('%s\n' % tweet)
        idx += 1
out4.close()


rest = './dataset/removed_data.txt'
idx = 0
rest_data = open(rest, 'r').readlines()
for line in rest_data:
    record = line.strip().split('\t')
    seq_num = int(record[0])
    try:
        tweet = record[1]
        out5 = open('./topics/removed_tweets/' + str(idx) + '.txt', 'w+')
        out5.write('%s\n' % tweet)
        out5.close()
        idx += 1
    except:
        continue

