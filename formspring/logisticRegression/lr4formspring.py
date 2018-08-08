import numpy as np
import os
from os.path import isfile, join
from os import listdir

from six.moves import cPickle as pickle
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TweetTokenizer
# from stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import enchant
d = enchant.Dict("en_US")

tokenizer = RegexpTokenizer(r'\w+')
# en_stop = get_stop_words('en')
stop_word = stopwords.words('english')
p_stemmer = PorterStemmer()


def remove_non_ascii_characters(string_in):
    stripped = [c for c in string_in if 0 < ord(c) < 127]
    return ''.join(stripped)


# Preprocess formspring data
formspringFile = '../labeledFormspring.txt'
data = open(formspringFile, 'r').readlines()
document = []
labels = []
for line in data:
    tmp = line.strip().split('\t')
    question = remove_non_ascii_characters(tmp[0].lower())
    answer = remove_non_ascii_characters(tmp[1].lower())
    severity = int(tmp[2])
    if severity==0:
        continue

    tweet = question+ "\t"+ answer
    # NPL Preprocessing
    tokens = tokenizer.tokenize(tweet)
    # print tokens
    stopped_tokens = [i for i in tokens if (i not in stop_word)]
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    clean_tweet = ("".join([" " + i for i in stemmed_tokens])).strip()
    document.append(clean_tweet)
    if severity<=3 and severity>0:
        labels.append(0)
    elif severity<7:
        labels.append(1)
    else:
        labels.append(2)

# Preprocess Sarahah data
file = '/home/yue/Public/Python/sarahah-feature/dataset/original_data.txt'
sarahahData = open(file, 'r').readlines()
sarahahDoc = []
sarahahCleanDoc = []
indexes = []
for line in sarahahData:
    record = line.strip().split('\t')
    idx = int(record[0])
    text = remove_non_ascii_characters(record[1].lower())
    try:
        comment = remove_non_ascii_characters(record[2].lower())
    except:
        print idx
        comment = ""
    tweet = text+ "\t"+ comment
    tokens = tokenizer.tokenize(tweet)
    stopped_tokens = [i for i in tokens if (i not in stop_word)
                      and (i not in ['sarahah', 'samanmm', 'sarahahcom', 'com'])
                      and (d.check(i))]  # and (d.check(i) and len(i) > 1
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    clean_tweet = ("".join([" " + i for i in stemmed_tokens])).strip()
    # print clean_tweet
    if len(clean_tweet) == 0:
        continue
    sarahahCleanDoc.append(clean_tweet)
    sarahahDoc.append(tweet)
    indexes.append(idx)


######
all_documents = document+ sarahahCleanDoc
no_features = 2000
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(all_documents)
tf_feature_names = tf_vectorizer.get_feature_names()

tf_dense = tf.toarray()
train_feature = tf_dense[0:len(document), :]
test_feature = tf_dense[len(document):, :]
target = np.array(labels)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(train_feature, target)

y_pred = logreg.predict(test_feature)

output = 'labeledSarahahText.txt'
out = open(output, 'w+')

output1 = 'labeledSarahahComment.txt'
out1 = open(output1, 'w+')

idx = 0
for text in sarahahDoc:
    tmp = text.split('\t')
    body = tmp[0]
    severity = y_pred[idx]

    idx += 1
    out.write('%s\t%d\n' % (body, severity))
    try:
        comment = tmp[1]
        if (len(comment)>1):
            out1.write('%s\t%d\n' % (comment, severity))
    except:
        continue
out.close()
out1.close()

