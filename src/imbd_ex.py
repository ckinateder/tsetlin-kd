import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from time import time
from typing import Union, Tuple

def prepare_imdb_data(
    max_ngram: int = 2,
    num_words: int = 5000,
    index_from: int = 2,
    features: int = 5000
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Returns the IMDB dataset in a format that can be used by the Tsetlin Machine.
    Returns:
        X_train: np.ndarray: The training data.
        Y_train: np.ndarray: The training labels.
        X_test: np.ndarray: The testing data.
        Y_test: np.ndarray: The testing labels.
    """

    print("Downloading dataset...")

    train,test = keras.datasets.imdb.load_data(num_words=num_words, index_from=index_from)

    train_x,train_y = train
    test_x,test_y = test

    word_to_id = keras.datasets.imdb.get_word_index()
    word_to_id = {k:(v+index_from) for k,v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2

    print("Producing bit representation...")

    # Produce N-grams

    id_to_word = {value:key for key,value in word_to_id.items()}

    vocabulary = {}
    for i in range(train_y.shape[0]):
        terms = []
        for word_id in train_x[i]:
            terms.append(id_to_word[word_id])
        
        for N in range(1,max_ngram+1):
            grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
            for gram in grams:
                phrase = " ".join(gram)
                
                if phrase in vocabulary:
                    vocabulary[phrase] += 1
                else:
                    vocabulary[phrase] = 1

    # Assign a bit position to each N-gram (minimum frequency 10) 

    phrase_bit_nr = {}
    bit_nr_phrase = {}
    bit_nr = 0
    for phrase in vocabulary.keys():
        if vocabulary[phrase] < 10:
            continue

        phrase_bit_nr[phrase] = bit_nr
        bit_nr_phrase[bit_nr] = phrase
        bit_nr += 1

    # Create bit representation
    X_train = np.zeros((train_y.shape[0], len(phrase_bit_nr)), dtype=np.uint32)
    Y_train = np.zeros(train_y.shape[0], dtype=np.uint32)
    for i in range(train_y.shape[0]):
        terms = []
        for word_id in train_x[i]:
            terms.append(id_to_word[word_id])

        for N in range(1,max_ngram+1):
            grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
            for gram in grams:
                phrase = " ".join(gram)
                if phrase in phrase_bit_nr:
                    X_train[i,phrase_bit_nr[phrase]] = 1

        Y_train[i] = train_y[i]

    X_test = np.zeros((test_y.shape[0], len(phrase_bit_nr)), dtype=np.uint32)
    Y_test = np.zeros(test_y.shape[0], dtype=np.uint32)

    for i in range(test_y.shape[0]):
        terms = []
        for word_id in test_x[i]:
            terms.append(id_to_word[word_id])

        for N in range(1,max_ngram+1):
            grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
            for gram in grams:
                phrase = " ".join(gram)
                if phrase in phrase_bit_nr:
                    X_test[i,phrase_bit_nr[phrase]] = 1				

        Y_test[i] = test_y[i]

    print("Selecting features...")

    SKB = SelectKBest(chi2, k=features)
    SKB.fit(X_train, Y_train)

    selected_features = SKB.get_support(indices=True)
    X_train = SKB.transform(X_train)
    X_test = SKB.transform(X_test)

    return (X_train, Y_train), (X_test, Y_test)
