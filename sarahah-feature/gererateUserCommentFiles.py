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
    try:
        comment = record[2]
    except:
        comment = ''
    stem_dict[seq_num] = (tweet, comment)

#
# topic0_file = '/home/yue/Public/C_program/cl2_project/SeededLDA/data/sarahah_data' \
#               '/new_result3/topics/topic0.txt'
# #'topics/explicit_se*_topic.txt'
# topic1_file = '/home/yue/Public/C_program/cl2_project/SeededLDA/data/sarahah_data' \
#               '/new_result3/topics/topic1.txt'
# #'topics/haterAndAbuse_topic.txt'
# topic2_file = '/home/yue/Public/C_program/cl2_project/SeededLDA/data/sarahah_data' \
#               '/new_result3/subtopic2/topics/topic2.txt'
# #'topics/flirting_topic.txt'
# topic3_file = '/home/yue/Public/C_program/cl2_project/SeededLDA/data/sarahah_data' \
#               '/new_result3/subtopic2/topics/topic3.txt'
#
# nonBully_file1 = '/home/yue/Public/C_program/cl2_project/SeededLDA/data/sarahah_data' \
#                  '/new_result3/subtopic2/topics/rest_topic.txt'
# rest = './dataset/removed_data.txt'
#
# topic0_seqList = []
# topic1_seqList = []
#
# output0 = './userComments/comments0.txt'
# out0 = open(output0, 'w+')
# topic0_data = open(topic0_file, 'r').readlines()
# idx = 0
# for line in topic0_data:
#     record = line.strip().split('\t')
#     seq_num = int(record[0])
#     value = stem_dict[seq_num]
#     comment = value[1]
#     if len(comment) != 0:
#         # out0 = open('./userComments/comment0/'+ str(idx)+'.txt', 'w+')
#         # out0.write('%s\n' % comment)
#         # out0.close()
#         # idx += 1
#         out0.write('%d\t%s\n' % (seq_num, comment))
#         # idx += 1
# out0.close()
#
#
# output1 = './userComments/comments1.txt'
# out1 = open(output1, 'w+')
# topic1_data = open(topic1_file, 'r').readlines()
# idx = 0
# for line in topic1_data:
#     record = line.strip().split('\t')
#     seq_num = int(record[0])
#     value = stem_dict[seq_num]
#     comment = value[1]
#     if len(comment) != 0:
#         # out1 = open('./userComments/comment1/' + str(idx)+'.txt', 'w+')
#         # out1.write('%s\n' % comment)
#         # out1.close()
#         # idx += 1
#         out1.write('%d\t%s\n' % (seq_num, comment))
#         # idx += 1
# out1.close()
#
# output2 = './userComments/comments2.txt'
# out2 = open(output2, 'w+')
# topic2_data = open(topic2_file, 'r').readlines()
# idx = 0
# for line in topic2_data:
#     record = line.strip().split('\t')
#     seq_num = int(record[0])
#     value = stem_dict[seq_num]
#     comment = value[1]
#     if len(comment) != 0:
#         # out2 = open('./userComments/comment2/'+str(idx)+'.txt', 'w+')
#         # out2.write('%s\n' % comment)
#         # out2.close()
#         # idx += 1
#         out2.write('%d\t%s\n' % (seq_num, comment))
#         # idx += 1
# out2.close()
#
# output3 = './userComments/comments3.txt'
# out3 = open(output3, 'w+')
# topic3_data = open(topic3_file, 'r').readlines()
# idx = 0
# for line in topic3_data:
#     record = line.strip().split('\t')
#     seq_num = int(record[0])
#     value = stem_dict[seq_num]
#     comment = value[1]
#     if len(comment) != 0:
#         # out3 = open('./userComments/comment3/'+str(idx)+'.txt', 'w+')
#         # out3.write('%s\n' % comment)
#         # out3.close()
#         # idx += 1
#         out3.write('%d\t%s\n' % (seq_num, comment))
#         # idx += 1
# out3.close()
#
#
# nonBully_data1 = open(nonBully_file1, 'r').readlines()
# # nonBully_data2 = open(nonBully_file2, 'r').readlines()
# # nonBully_data = nonBully_data1+ nonBully_data2
# out4 = open('./userComments/comments4.txt', 'w+')
# idx = 0
# for line in nonBully_data1:
#     record = line.strip().split('\t')
#     seq_num = int(record[0])
#     value = stem_dict[seq_num]
#     comment = value[1]
#     if len(comment) != 0:
#         # out4 = open('./userComments/comment4/'+str(idx)+'.txt', 'w+')
#         # out4.write('%s\n' % comment)
#         # idx += 1
#         out4.write('%d\t%s\n' % (seq_num, comment))
# out4.close()
#
#
# # idx = 0
# rest_data = open(rest, 'r').readlines()
# out5 = open('./userComments/comments5.txt', 'w+')
# for line in rest_data:
#     record = line.strip().split('\t')
#     seq_num = int(record[0])
#     try:
#         comment = record[2]
#         # out5 = open('./userComments/comment5/' + str(idx) + '.txt', 'w+')
#         # out5.write('%s\n' % comment)
#         # out5.close()
#         # idx += 1
#         out5.write('%d\t%s\n' % (seq_num, comment))
#     except:
#         continue
# out5.close()


# targeting_women_file = '/home/yue/Public/C_program/cl2_project/SeededLDA/data/' \
#                        'sarahah_data/new_result3/subtopic0/topics/topic0.txt'
# targeting_men_file = '/home/yue/Public/C_program/cl2_project/SeededLDA/data/' \
#                        'sarahah_data/new_result3/subtopic0/topics/topic1.txt'

targeting_women_file = './topics/topic0.txt'
targeting_men_file = './topics/topic1.txt'

men_data = open(targeting_men_file, 'r').readlines()
women_data = open(targeting_women_file, 'r').readlines()

idx = 0
for line in men_data:
    record = line.strip().split('\t')
    seq_num = int(record[0])
    value = stem_dict[seq_num]
    tweet = value[0]
    comment = value[1]
    if len(comment) > 0:
        out = open('./userComments/men/'+str(idx)+'.txt', 'w+')
        out.write('%s\n' % comment)
        out.close()
        idx += 1
out.close()

idx = 0
for line in women_data:
    record = line.strip().split('\t')
    seq_num = int(record[0])
    value = stem_dict[seq_num]
    tweet = value[0]
    comment = value[1]

    if len(comment) > 0:
        out = open('./userComments/women/'+str(idx)+'.txt', 'w+')
        out.write('%s\n' % comment)
        out.close()
        idx += 1
out.close()





