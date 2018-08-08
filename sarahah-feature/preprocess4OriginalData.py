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


tokenizer = RegexpTokenizer(r'\w+')  # r'[a-zA-Z]+'
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
# create English stop words list
en_stop = get_stop_words('en')
stop_word = stopwords.words('english')
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
d = enchant.Dict("en_US")


# set1 = ['sexi', 'ass', 'bitch', 'oral', 'anal', 'gay', 'hot', 'fantasi', 'stalk', 'nasti', 'fuck', 'boob', 'dick',
#         'nake', 'suck', 'lip', 'size', 'lick', 'tongu', 'turn', 'booti', 'thigh', 'bra', 'bed', 'horni', 'seduct',
#         'ball', 'hoe', 'virgin', 'lesbian', 'bite', 'butt', 'straight', 'leg', 'beast', 'fluid', 'chocolati',
#         'syrup', 'vagina', 'threesom', 'belli', 'homosexu']

# set2 = ['crush', 'date', 'dreams', 'friend', 'miss', 'babe', 'sweeti', 'candi', 'look', 'pie', 'appeal',
#         'crave', 'propos', 'hit', 'cheek', 'feel', 'romanc', 'poetri', 'hang', 'desir', 'pleasur', 'bomb',
#         'cute', 'eye', 'hug', 'chick', 'marri', 'love', 'babi', 'exchang', 'coffe', 'video']

# set3 = ['shoot', 'kick', 'fat', 'bullshit', 'threat', 'fight', 'death', 'rude', 'ruin', 'shit', 'ugli', 'abus',
#         'ego', 'sad', 'mad', 'cheat', 'trash', 'pain', 'tear', 'cri', 'trap', 'annoy', 'ex', 'chuck', 'fragil',
#         'punch', 'slap', 'betray', 'harm', 'loath', 'fire', 'blunt', 'emot', 'breakup', 'annoy']

# set4 = ['life', 'hard', 'kiss', 'beauti', 'toxic', 'hater', 'asshol', 'hate', 'pretti', 'sex', 'weak',
#         'slap', 'screw', 'die']
# 'sexchat', 'seek', 'cutie', 'pie', 'bf', 'gf', 'husband', 'wife', 'position',
        #'blowjob', 'pic', 'disgust'

bullyWords = list(set(set1+ set2+ set3+ set4))

deleteList = ['#sarahah@sarahah_com', '#sarahah']

data_dir = './dataset'
englishData_file = join(data_dir, 'EnglishData.txt')
nonEnglishData_file = join(data_dir, 'NonEnglishData.txt')
userSpecificData_file = join(data_dir, 'UserSpecificData.txt')


fileList = [englishData_file, nonEnglishData_file, userSpecificData_file]

src_output = join(data_dir, 'src_data.txt')
src_out = open(src_output, 'w+')

stem_output = join(data_dir, 'stem_data.txt')
stem_out = open(stem_output, 'w+')

original_output = join(data_dir, 'original_data.txt')
original_out = open(original_output, 'w+')

remove_output = join(data_dir, 'removed_data.txt')
remove_out = open(remove_output, 'w+')

seq_num = 0
for file in fileList:
    english_data = open(file, 'r').readlines()

    idx = -1
    for line in english_data:
        idx += 1
        if idx == 0:
            continue
        record = line.strip().split('\t')

        db_id = int(record[0])
        tweet = record[1].lower()
        try:
            comment = record[2].lower()
        except:
            comment = ''

        tmp_tweet = tweet
        tmp_comment = comment

        if len(tweet) == 0 or tweet == '' or tweet == 'sarahah with':
            continue

        tokens = tokenizer.tokenize(tweet)
        stopped_tokens = [i for i in tokens if (i not in en_stop) and (i not in stop_word)
                          and (i not in ['sarahah', 'samanmm', 'sarahahcom', 'com'])
                          and (d.check(i))]  # and (d.check(i) and len(i) > 1
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        clean_tweet = ("".join([" " + i for i in stemmed_tokens])).strip()
        tweet = ("".join([" " + i for i in stopped_tokens])).strip()

        check_flag = 0
        for w in stemmed_tokens:
            # if str(w) in bullyWords:
            #     check_flag = 1
            #     break
        # check_flag = 1
        # if len(stemmed_tokens)<=1:
        #     continue

        comment_words = comment.split(' ')
        clean_text = []
        for w in comment_words:
            flag = 0
            if (w not in deleteList) and ('https://' not in w) and (w != 'rt') and ('@' not in w):
                for deleteWord in deleteList:
                    try:
                        idx = w.index(deleteWord)
                    except:
                        continue
                    w = w.replace(deleteWord, '')
                    clean_text.append(w)
                    flag = 1
                if flag == 0:
                    clean_text.append(w)
        clean_text = ("".join([" " + i for i in clean_text])).strip()

        tokens = tokenizer.tokenize(clean_text)
        stopped_tokens = [i for i in tokens if (i not in en_stop) and (i not in stop_word)]
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        clean_comment = ("".join([" " + i for i in stemmed_tokens])).strip()
        comment = ("".join([" " + i for i in stopped_tokens])).strip()

        if check_flag == 0:
            remove_out.write('%d\t%s\t%s\n' % (seq_num, tweet, comment))
            continue

        src_out.write('%d\t%s\t%s\n' % (seq_num, tweet, comment))
        stem_out.write('%d\t%s\t%s\n' % (seq_num, clean_tweet, clean_comment))

        original_out.write('%d\t%s\t%s\n' % (seq_num, tmp_tweet, tmp_comment))

        seq_num += 1

src_out.close()
stem_out.close()
original_out.close()
remove_out.close()

