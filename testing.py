from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.datasets import fetch_20newsgroups

import sys

from nltk import stem
from nltk.corpus import stopwords 

from numpy import mean

from collections import namedtuple

import datetime

import re

import csv

from colorama import Fore

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

"""
This module contains classes for testing a basic text classification
pipeline. The classifier has been hardcoded as SVC, but this can
quite easily be changed.

Classes are glorified functions; configured at __init__ and then
called (__call__) to perform an action.

Run as __main__ to test with 20newsgroups.

valid config values are:

    preprocessing:
        lower - lowercase the text
        nonum - remove all numbers
        stem - stem words
        stopwords - remove stopwords

    vectorization:
        tfidf - tfidf-transform matrix after vectorizing
        bigram - vectorize with bigrams
        trigram - vectorize with trigrams 

    classification:
        tune - tune C parameter between 0.01 and 10
"""
 
# ================================================

def fallout(y_true, y_pred):
    """
    Fallout loss scorer, somehow not implemented in sklearn
    """
    mat = confusion_matrix(y_true, y_pred)

    import csv
    tn, fp, fn, tp = mat.ravel() 
    fo = fp / (tn + fp)
    return(fo)

fallout_loss = make_scorer(fallout)

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

class Preprocessor():
    """
    A text preprocessor, performs transformations of text according to
    what is specified in config

    Attributes:
        config: A list specifying the preprocessing steps
    """ 
    def __init__(self, config):
        """
        Constructs a list of preprocessing steps specified in config
        Steps are internal methods specified below
        """
        self.steps = []

        self.stemmer = stem.PorterStemmer()
        self.swords = stopwords.words('english')

        if "lower" in config:
            self.steps.append(self._lower)
        if "nonum" in config:
            self.steps.append(self._nonum)
        if "stem" in config:
            self.steps.append(self._stem)
        if "stopwords" in config:
            self.steps.append(self._stopwords)

    def __call__(self, text):
        """
        Performs preprocessing of the text, iterating over steps in self.steps
        """
        for f in self.steps:
            text = f(text)
        return(text)

    def _lower(self, text):
        return(text.lower())

    def _nonum(self, text):
        return(re.sub('[0-9]+','*number*',text))

    def _stem(self, text):
        text = [self.stemmer.stem(w) for w in text.split()]
        return(' '.join(text))

    def _stopwords(self, text):
        text = [w for w in text.split() if w.lower() not in self.swords]
        return(' '.join(text))

# ================================================

class Procedure():
    """
    A text-classification testing procedure that is first specified, and then
    called on a text corpus of documents.
    
    Attributes:
        config: A list specifying the procedure to be tested 
    """ 
    def __init__(self, config, folds = 10, verbose = True):
        """
        Initializes the procedure with a configuration
        """
        self.config = config
        self.folds = folds 
        self.verbose = verbose

        self.preprocessor = Preprocessor(config)

        if "bigram" in self.config:
            ngram_range = (1,2)
        elif "trigram" in self.config:
            ngram_range = (1,3)
        else:
            ngram_range = (1,1)

        self.vectorizer = CountVectorizer(lowercase = False,
                                          ngram_range = ngram_range)

    def __call__(self, docs, y):
        """
        Test the procedure on docs, returning a dict of scores
        """

        if self.verbose:
            print(('*'*10) + f"{Fore.YELLOW}\nRunning model:\n\"{'_'.join(self.config)}\"")

        m1 = datetime.datetime.now()

        scoring = {'fallout': fallout_loss, 'recall':'recall',
                'precision':'precision', 'f1':'f1'}

        docs = [self.preprocessor(d) for d in docs]

        m2 = datetime.datetime.now()

        mat = self.vectorizer.fit_transform(docs)

        if "tfidf" in self.config:
            tfidft = TfidfTransformer()
            mat = tfidft.fit_transform(mat)
        else:
            pass
        
        m3 = datetime.datetime.now()

        if "tune" in self.config:
            self.C = self._CSearch(mat,y) 
        else:
            self.C = 1

        m4 = datetime.datetime.now()
        
        self.classifier = SVC(C = self.C, kernel = 'linear')

        self.scores = cross_validate(self.classifier,mat,y,
                                cv = self.folds, scoring = scoring, n_jobs = 7,
                                return_train_score = False)

        m5 = datetime.datetime.now()
        self.times = {'time_total': m5 - m1,
                      'time_preproc': m2 - m1,
                      'time_vect' : m3 - m2,
                      'time_tuning' : m4 - m3,
                      'time_crossval': m5 - m4}

        if self.verbose:
            results = {'Time elapsed' : self.times['time_total'].total_seconds(),
                       'Features' : len(self.vectorizer.vocabulary_),
                       'Mean F1' :  mean(self.scores['test_f1'])}
            resstring = '\n'.join([f'{n}: {v}' for n,v in results.items()])
            print(f'\n{Fore.YELLOW}Results:\n{Fore.GREEN}{resstring}{Fore.RESET}\n')


        return(self.scores)

    def _CSearch(self,x,y):
        gs = GridSearchCV(SVC(), n_jobs = 7, scoring = 'f1', cv = self.folds,
                          param_grid = {'C' : [0.001,0.01,0.1,1,10],
                                        'kernel':['linear']})
        gs.fit(x,y)
        return(gs.best_params_['C'])
    
    def dump(self):
        """
        Output a dictionary version of itself, that summarizes the
        characteristics and performance of the procedure
        """
        rep = self.scores
        rep.update({'config':self.config})
        rep.update({'C':self.C})
        rep.update({'features':len(self.vectorizer.vocabulary_)})
        rep.update({k:v.total_seconds() for k,v in self.times.items()})
        rep.update({'meanF1': mean(self.scores['test_f1'])})

        return(rep)

# ================================================

if __name__ == '__main__':
    config = sys.argv[1:]
    procedure = Procedure(config)

    twentyng = fetch_20newsgroups(categories = ["misc.forsale",
                                            "soc.religion.christian"])

    scores = procedure(twentyng.data, twentyng.target)

