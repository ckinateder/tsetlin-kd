from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from time import time
from typing import Union, Tuple
import h5py
import os
from abc import ABC, abstractmethod
from torchvision.datasets import KMNIST
from torchvision import transforms
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

def scale(X: np.ndarray, original_shape: tuple, scale_factor: int) -> np.ndarray:
    """
    Resizes each sample in X by scaling the original shape dimensions by scale_factor.
    
    Args:
        X: 2D array of shape (samples, features)
        original_shape: Tuple representing the original dimensions of each sample
        scale_factor: Factor to scale each dimension by
        
    Returns:
        Scaled 2D array of shape (samples, new_feature_count)
    """
    assert np.prod(original_shape) == X.shape[1], "Original shape doesn't match feature count"
    
    scaled_data = []
    
    for sample in X:
        img = sample.reshape(original_shape)
        for axis in range(len(original_shape)):
            img = np.repeat(img, scale_factor, axis=axis)
        scaled_data.append(img.flatten())
    
    return np.array(scaled_data)

class Dataset(ABC):
    def __init__(self, **kwargs):
        self._load(**kwargs)
    
    @abstractmethod
    def _load(self, **kwargs):
        raise NotImplementedError

    def get_data(self):
        return self.X_train, self.Y_train, self.X_test, self.Y_test

    def get_data_train(self):
        return self.get_data()[:2]

    def get_data_test(self):
        return self.get_data()[2:]

    def validate_lengths(self):
        assert len(self.X_train) == len(self.Y_train), "Training data length mismatch"
        assert len(self.X_test) == len(self.Y_test), "Testing data length mismatch"

class ImageDataset(Dataset):
    def __init__(self, scale_factor: int = 1, **kwargs):
        self.scale_factor = scale_factor
        self.shape = None
        super().__init__(**kwargs)
        
    def scale(self, X: np.ndarray):
        if self.shape is None:
            raise ValueError("Shape not set")
        return scale(X, self.shape, self.scale_factor)
    
    def get_data(self):
        if self.scale_factor != 1:
            return self.scale(self.X_train), self.Y_train, self.scale(self.X_test), self.Y_test
        return super().get_data()

class IMDBDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load(self, **kwargs):
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = prepare_imdb_data()

class MNISTDataset(ImageDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shape = (28, 28)
    def _load(self, booleanize_threshold: int = 75):
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = mnist.load_data()

        self.X_train = np.where(self.X_train > booleanize_threshold, 1, 0)
        self.X_test = np.where(self.X_test > booleanize_threshold, 1, 0)

        self.X_train = self.X_train.reshape(self.X_train.shape[0], 28*28)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 28*28)
        
class FashionMNISTDataset(ImageDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shape = (28, 28)
    def _load(self, booleanize_threshold: int = 75):
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = fashion_mnist.load_data()
        
        self.X_train = np.where(self.X_train > booleanize_threshold, 1, 0)
        self.X_test = np.where(self.X_test > booleanize_threshold, 1, 0)

        self.X_train = self.X_train.reshape(self.X_train.shape[0], 28*28)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 28*28)

class KMNISTDataset(ImageDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shape = (28, 28)

    def _load(self, booleanize_threshold: float = 75):
        train = KMNIST(root="data", download=True, train=True, transform=transforms.ToTensor())
        test = KMNIST(root="data", download=True, train=False, transform=transforms.ToTensor())
        self.X_train, self.Y_train = train.data.numpy(), train.targets.numpy()
        self.X_test, self.Y_test = test.data.numpy(), test.targets.numpy()

        self.X_train = self.X_train.reshape(self.X_train.shape[0], 28*28)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 28*28)

        self.X_train = np.where(self.X_train > booleanize_threshold, 1, 0)
        self.X_test = np.where(self.X_test > booleanize_threshold, 1, 0)
        
class MNIST3DDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load(self, booleanize_threshold: float = 0.3):
        with h5py.File(os.path.join("data", "mnist3d.h5"), "r") as hf:
            self.X_train = hf["X_train"][:]
            self.Y_train = hf["y_train"][:]    
            self.X_test = hf["X_test"][:]  
            self.Y_test = hf["y_test"][:]  
        
        self.X_train = np.where(self.X_train > booleanize_threshold, 1, 0)
        self.X_test = np.where(self.X_test > booleanize_threshold, 1, 0)

        self.X_train = self.X_train.reshape(self.X_train.shape[0], 16*16*16)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 16*16*16)


if __name__ == "__main__":
    dataset = IMDBDataset()
    data = dataset.get_data()

    print(data[0].shape)