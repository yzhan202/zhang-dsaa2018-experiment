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


data_dir = './dataset'
file = join(data_dir, 'tweetWithGender.txt')

genders = ['female', 'male']


stem_dict = {}
stem_file = './dataset/stem_data.txt'
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


data = open(file, 'r').readlines()
gender_dict = {}
for line in data:
    record = line.strip().split('\t')
    id = int(record[0])
    gender = record[2]
    # print gender
    gender_id = genders.index(gender)

    gender_dict[id] = gender_id


topic0_file = '/home/yue/Public/C_program/cl2_project/SeededLDA/data/' \
              'sarahah_data/new_result3/topics/topic0.txt'

topic0_id_list = []
data = open(topic0_file, 'r').readlines()
for line in data:
    record = line.strip().split('\t')
    id = int(record[0])
    topic0_id_list.append(id)


female_file = join('./genders', 'female.txt')
male_file = join('./genders', 'male.txt')

female_out = open(female_file, 'w+')
male_out = open(male_file, 'w+')

for i in range(len(topic0_id_list)):
    id = topic0_id_list[i]
    if id in gender_dict:
        gender_id = gender_dict[id]
        value = stem_dict[id]
        tweet = value[0]
        comment = value[1]
        if gender_id == 0:
            female_out.write('%d\t%s\n' % (id, tweet))
        else:
            male_out.write('%d\t%s\n' % (id, tweet))

female_out.close()
male_out.close()






