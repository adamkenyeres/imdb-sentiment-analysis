"""Module containing all vectorizers used my NnModel."""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.feature_extraction.text import CountVectorizer


class Vectorizer:
    """Base class for vectorizing classes."""

    def __init__(self, name, feature_number = 10000):
        self.name = name
        self.feature_number = feature_number

    def transform(self, arr):
        """Warning had been disabled as this class should not be inicialized."""
        arr =  self.vectorize_set(arr) # pylint: disable=E1111
        arr = np.array(arr).astype('float32')
        return arr.reshape(len(arr), -1)

    def get_name(self):
        return self.name

    def vectorize_set(self, data):
        pass

    def fit_transform(self, data):
        pass


class TextVectorizer(Vectorizer):
    """Text vectorizer contains an instance of TextVectorization from TF."""

    def __init__(self, name, feature_number=10000, length=500):
        Vectorizer.__init__(self, name, feature_number)
        self.length = length

    def vectorize(self, text):
        text = tf.expand_dims(text, -1)
        return self.vectorizer(text)

    def vectorize_set(self, data):
        data =  [self.vectorize(t) for t in data]
        return data

    def fit_transform(self, data):
        print(f'Training {self.name}')
        data = np.array(data)
        self.vectorizer = TextVectorization(
            max_tokens=self.feature_number,
            output_mode='int',
            output_sequence_length=self.length)

        self.vectorizer.adapt(data)

        return self.transform(data)


class CountVectorization(Vectorizer):
    '''
    Count vectorizer contains an instance of CountVectorizer from sklearn'''
    def __init__(self, name, feature_number = 10000):
        Vectorizer.__init__(self, name, feature_number)

    def vectorize(self, text):
        return self.vectorizer.transform(text)

    def vectorize_set(self, data):
        return self.vectorize(data).toarray()

    def fit_transform(self, data):
        print(f'Training {self.name}')

        self.vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                          max_features=self.feature_number)

        self.vectorizer.fit(data)
        return self.transform(data)
