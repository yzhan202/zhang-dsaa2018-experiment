import numpy as np
import os
from os.path import isfile, join
from six.moves import cPickle as pickle
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TweetTokenizer
from stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def preprocess_training_data(file):
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    header = ['OriginalText', 'ProcessedText', 'Severity']

    target = []
    documents = []

    data = open(file, 'r').readlines()
    start = 0
    for line in data:
        record = line.strip().split('\t')
        if start == 0:
            start = 1
            continue
        text = record[0]
        clean_text = record[1]
        try:
            label = int(record[2])
        except:
            print('No label')
            continue

        documents.append(clean_text)
        if label>=0 and label<=3:
            class_tag = 0
        elif label>=4 and label<=6:
            class_tag = 1
        else:
            class_tag = 2
        target.append(class_tag)

    target = np.array(target)

    no_features = 2000
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()
    tf_dense = tf.toarray()

    for i in range(20):
        print tf_feature_names[i]

    return tf_dense, target


def preprocess(train_file, test_file):

    #
    # training dataset
    #
    header = ['OriginalText', 'ProcessedText', 'Severity']

    target = []
    train_documents = []

    data = open(train_file, 'r').readlines()
    start = 0
    for line in data:
        record = line.strip().split('\t')
        if start == 0:
            start = 1
            continue
        text = record[0]
        clean_text = record[1]
        try:
            label = int(record[2])
        except:
            print('No label')
            continue

        train_documents.append(clean_text)
        if label >= 0 and label <= 3:
            class_tag = 0
        elif label >= 4 and label <= 6:
            class_tag = 1
        else:
            class_tag = 2
        target.append(class_tag)

    target = np.array(target)

    #
    # test dataset
    #
    import enchant
    d = enchant.Dict("en_US")

    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    stop_word = stopwords.words('english')
    p_stemmer = PorterStemmer()

    # print stop_word

    test_documents = []
    indexes = []

    data = open(test_file, 'r').readlines()
    idx = -1
    for line in data:
        idx += 1
        if idx == 0:
            continue
        tweet = line.strip().lower()
        if tweet == 'sarahah with':
            continue

        tokens = tokenizer.tokenize(tweet)
        # print tokens
        stopped_tokens = [i for i in tokens if (i not in en_stop) and (i not in stop_word)
                          and (i not in ['sarahah', 'samanmm', 'sarahahcom', 'com'])
                          and (d.check(i))]  # and (d.check(i) and len(i) > 1
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        clean_tweet = ("".join([" " + i for i in stemmed_tokens])).strip()
        # print clean_tweet
        if len(clean_tweet) == 0:
            print line
            continue
        test_documents.append(clean_tweet)
        indexes.append(idx)

    all_documents = train_documents+ test_documents

    no_features = 2000
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(all_documents)
    tf_feature_names = tf_vectorizer.get_feature_names()

    # for i in range(20):
    #     print tf_feature_names[i]

    tf_dense = tf.toarray()
    train_feature = tf_dense[0:len(train_documents),:]
    test_feature = tf_dense[len(train_documents):,:]

    return train_feature, test_feature, target, test_documents


def train(feature, target):
    logreg = LogisticRegression()
    logreg.fit(feature, target)

    return logreg


def predict(logreg, test_feature):
    y_pred = logreg.predict(test_feature)
    y_proba = np.array(logreg.predict_proba(test_feature))
    return y_pred, y_proba


def main():
    train_file = './formspring_data_logistic_regression.txt'
    test_file = './Sarahah_data_logistic_regression_test.txt'

    train_feature, test_feature, target, test_documents = preprocess(train_file, test_file)

    # max_label = np.max(target)
    # min_label = np.min(target)
    # mean_label = np.mean(target)
    # median_label = np.median(target)
    #
    # print max_label, min_label, mean_label, median_label
    # for i in range(len(target)):
    #     print target[i]

    logreg = train(train_feature, target)

    y_pred, y_proba = predict(logreg, test_feature)

    # for i in range(40):
    #     print y_pred[i], target[i]
    output = './output.txt'
    out = open(output, 'w+')
    for i in range(len(y_pred)):
        # print idx
        tweet = test_documents[i]
        prediction = y_pred[i]
        out.write('%d\t%s\n' % (prediction, tweet))

if __name__ == "__main__":
    main()