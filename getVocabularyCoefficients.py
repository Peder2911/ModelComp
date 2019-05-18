
def getVocabCoefficients(classifier,vectorizer):
    coefficients = classifier.coef_.toarray().tolist()[0]

    vocabulary = [*vectorizer.vocabulary_.items()]
    vocabulary.sort(key = lambda entry: entry[1])
    vocabulary = [w for w,_ in vocabulary]

    vocab_coef = [*zip(vocabulary, coefficients)]
    vocab_coef.sort(key = lambda entry: -entry[1])
    return(vocab_coef)


