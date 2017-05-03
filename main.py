import numpy as np
import scipy as sc
import os, re
import matplotlib.pyplot as plt
from prettyprint import pp
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, classification_report
from datetime import datetime as dt
from ipy_table import *
from string import punctuation, digits


data_path = '/Users/alexandre/Projects/TextClassificationBayes3/tok_vnexpress/'
stopwords_path = '/Users/alexandre/Projects/TextClassificationBayes3/vietnamese-stopwords-dash.txt'


#Classes are the folder names
class_names = os.listdir(data_path)
folders =  [data_path + folder + '/' for folder in os.listdir(data_path) if folder != ".DS_Store"]

if '.DS_Store' in class_names: del class_names[0]


#list the files of each class
files = {}

for folder, name in zip(folders, class_names):
    files[name] = [folder + f for f in os.listdir(folder)]
    
punctuation = re.compile(r'[-.?!,":;()|0-9]')
    
train_test_ratio = 0.9

def train_test_split(ratio, classes, files):
    """
    this method will split the input list of files to train and test sets.
    *Note: currently this method uses the simplest way an array can be split in two parts.
    Parameters
    ----------
    ratio: float
           ratio of total documents in each class assigned to the training set
    classes: list
             list of label classes
    files: dictionary
           a dictionary with list of files for each class
    
    Returns
    -------
    train_dic: dictionary
                a dictionary with lists of documents in the training set for each class
    test_dict: dictionary
                a dictionary with lists of documents in the testing set for each class
    """
    train_dict = {}
    test_dict = {}
    for cl in classes:
        train_cnt = int(ratio * len(files[cl]))
        train_dict[cl] = files[cl][:train_cnt]
        test_dict[cl] = files[cl][train_cnt:]
    return train_dict, test_dict


train_path, test_path = train_test_split(train_test_ratio, class_names, files)
#train_path, test_path, class_train, class_test = train_test_split(files, class_names, test_size=0.3, random_state=42)
stop_word = []

def loadStopWord(path):

    from string import punctuation, digits
    stop_word = []
    try:
        f = open(path)
        lines = [line.rstrip('\n') for line in open(path)]
        lines = [line.replace(' ', '_') for line in lines]
            
    finally:
        f.close()
    return lines

stop_word = loadStopWord(stopwords_path)


def cleanupText(path):
    """
    this method will read in a text file and try to cleanup its text.
    Parameters
    ----------
    path: path to the document file
    
    Returns
    -------
    text_cleaned: cleaned up raw text in the input file
    """
    
    text_cleaned = ''
    try:
        f = open(path)
        raw = f.read().lower()
        text =  raw   
        text_cleaned = re.sub(r'[-.?!,":;()/|0-9]','', text)
        
#        splitword = text_cleaned.split(" ")
#        
#        for word in splitword:
#            if word in stop_word:
#                splitword.remove(word)
#        text_cleaned = " ".join(splitword)
#        # print "\n Word count after:" + str(len(text_translated.split())) + "\n"
#        text_cleaned = ' '.join([word for word in text_cleaned.split(' ') if (word and len(word) > 1)])
    finally:
        f.close()
    return text_cleaned



#print(stop_word)
train_arr = []
test_arr = []
train_lbl = []
test_lbl = []


for cl in class_names:
    for path in train_path[cl]:
        train_arr.append(cleanupText(path))
        train_lbl.append(cl)
    for path in test_path[cl]:
        test_arr.append(cleanupText(path))
        test_lbl.append(cl)
        
#print len(train_arr)
#print len(test_arr)

#
vectorizer = CountVectorizer()
vectorizer.fit(train_arr)
train_mat = vectorizer.transform(train_arr)
#print train_mat.shape
#print train_mat
test_mat = vectorizer.transform(test_arr)
#print test_mat.shape

#tfidf = TfidfTransformer()
#tfidf.fit(train_mat)
#train_tfmat = tfidf.transform(train_mat)
##print train_tfmat.shape
##print train_tfmat
#test_tfmat = tfidf.transform(test_mat)
##print test_tfmat.shape


def testClassifier(x_train, y_train, x_test, y_test, clf):
    """
    this method will first train the classifier on the training data
    and will then test the trained classifier on test data.
    Finally it will report some metrics on the classifier performance.
    
    Parameters
    ----------
    x_train: np.ndarray
             train data matrix
    y_train: list
             train data label
    x_test: np.ndarray
            test data matrix
    y_test: list
            test data label
    clf: sklearn classifier object implementing fit() and predict() methods
    
    Returns
    -------
    metrics: list
             [training time, testing time, recall and precision for every class, macro-averaged F1 score]
    """
    #metrics = []
    start = dt.now()
    clf.fit(x_train, y_train)
    end = dt.now()
    print 'training time: ', (end - start)
    
    # add training time to metrics
    #metrics.append(end-start)
    
    start = dt.now()
    yhat = clf.predict(x_test)
    end = dt.now()
    print 'testing time: ', (end - start)
    
    # add testing time to metrics
    #metrics.append(end-start)
    
    print 'classification report: '
    #print classification_report(y_test, yhat)
    pp(classification_report(y_test, yhat))
    
    print 'f1 score'
    print f1_score(y_test, yhat, average='weighted')
    
    print 'accuracy score'




metrics_dict = {}
#'name', 'metrics'
#bnb = BernoulliNB()
#bnb_me = testClassifier(train_mat, train_lbl, test_mat, test_lbl, bnb)
#metrics_dict.update({1:bnb})
#bnb = BernoulliNB(alpha=0.001)
#bnb_me = testClassifier(train_mat, train_lbl, test_mat, test_lbl, bnb)
#metrics_dict.update({1:bnb})
#
#
#gnb = GaussianNB()
#gnb_me = testClassifier(train_tfmat.toarray(), train_lbl, test_tfmat.toarray(), test_lbl, gnb)
#metrics_dict.append({'name':'GaussianNB', 'metrics':gnb_me})
#alpha=[10,5]

#for a in alpha:
#    print "alpha= =" + str(a)
#    mnb = MultinomialNB(a)
#    mnb_me = testClassifier(train_tfmat.toarray(), train_lbl, test_tfmat.toarray(), test_lbl, mnb)
#    metrics_dict.update({a:mnb_me})
#
#mnb = MultinomialNB(alpha=0.001)
#k_fold = KFold(len(train_lbl), n_folds=10, shuffle=True, random_state=0)
#print cross_val_score(mnb, train_mat, train_lbl, cv=k_fold, n_jobs=1)

#mnb_me = testClassifier(train_mat, train_lbl, test_mat, test_lbl, mnb)
#metrics_dict.update({0.001:mnb_me})
