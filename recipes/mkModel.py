
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.metrics import confusion_matrix

import csv

from collections import namedtuple

import sys

"""
Make the top-performing model from the thesis

Include this script!
"""

# ================================================

def csvCorpus(f):
    rd = csv.reader(f.readlines())
    next(rd)

    txt = []
    val = []

    for l in rd:
        txt.append(l[0])
        val.append(int(l[1]))

    corpus = namedtuple('corpus',['data','target'])
    corpus.data = txt
    corpus.target = val

    return(corpus)

# ================================================
training_data_file, testing_data_file  = (sys.argv[1], sys.argv[2])

with open(training_data_file) as f:
    training_data = csvCorpus(f)

with open(testing_data_file) as f:
    testing_data = csvCorpus(f)

# ================================================

vectorizer = CountVectorizer(lowercase = False,
                             ngram_range = (1,1))

transformer = TfidfTransformer()

classifier = SVC(C = 1, kernel = 'linear')

# ================================================

training_matrix = vectorizer.fit_transform(training_data.data)
training_matrix = transformer.fit_transform(training_matrix)

classifier.fit(training_matrix,training_data.target)

# ================================================

testing_matrix = vectorizer.transform(testing_data.data)
testing_matrix = transformer.transform(testing_matrix)

prediction = classifier.predict(testing_matrix)

# ================================================

mat = confusion_matrix(testing_data.target, prediction)

def dumpConfMatrix(matrix,f):
    results = []
    for row in matrix.tolist():
        results.append({'Negatives':row[0],'Positives':row[1]})

    wr = csv.DictWriter(f,fieldnames = results[0].keys())
    wr.writeheader()

    for e in results:
        wr.writerow(e)

