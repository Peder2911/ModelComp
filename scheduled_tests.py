
import testing
import json
import sys

import logging

import numpy

"""
Run tests scheduled in a json file at argv[1], specifying the test name and the
configuration of the test procedure.

The tests are run on text data specified in argv[2].

Outputs lots of data about each of the procedures.
"""

SCHED_FILE = sys.argv[1]
DATA_FILE = sys.argv[2]

# ================================================

with open(SCHED_FILE) as f:
    schedule = json.load(f)

with open(DATA_FILE) as f:
    corpus = testing.csvCorpus(f)

# ================================================

procedures = {name:testing.Procedure(conf) for name,conf in schedule.items()}

for name,procedure in procedures.items():
    procedure(corpus.data,corpus.target)

results = {name:procedure.dump() for name,procedure in procedures.items()}

# ================================================
"""
Writeout
"""

def fixNpArrays(x):
    """
    To facilitate serialization
    """
    if type(x) == numpy.ndarray:
        x = x.tolist()
    else:
        pass
    return(x)

for name,modelResults in results.items():
    modelResults = {n:fixNpArrays(r) for n,r in modelResults.items()}
    with open('results/' + name + '.json','w') as f:
        json.dump(modelResults,f)

