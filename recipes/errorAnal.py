
ppSentences(sentences,y,x):
    for i,s in enumerate(errorSents):
        print('#'*38)
        print(f'{s} - pred: {prediction[err][i]} | actual: {actual[err][i]}')
        print('\n')

