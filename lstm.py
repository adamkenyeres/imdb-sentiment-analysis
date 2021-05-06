"""Module containing LSTM Implementation."""

from model import Model
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras import losses
from keras_visualizer import visualizer

class LSTM(Model):
    '''
    Base class for vectorizing classes.'''
    def __init__(self, name, vectorizer, features = 10000, epochs = 10):
        Model.__init__(self, name, vectorizer)
        self.features = features
        self.epochs = epochs
        self.model = None
        self.history = None

    def build(self):
        self.model = Sequential()
        self.model.add(layers.Embedding(self.features + 1, 16))
        self.model.add(layers.Bidirectional(layers.LSTM(
            64, return_sequences=True)))
        self.model.add(layers.Bidirectional(layers.LSTM(32)))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(units=64, activation='relu'))
        self.model.add(layers.Dense(1))

        self.model.summary()

        self.model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                optimizer='adam',
                metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

    def train(self, train_X, train_Y, val_X, val_Y):
        train_X = self.vectorizer.fit_transform(train_X)
        val_X = self.vectorizer.transform(val_X)

        self.history = self.model.fit(
            train_X,train_Y,
            validation_data=(val_X,val_Y),
            epochs=self.epochs,
            callbacks=[self.tensorboard])


    def evaulate(self, test_x, test_y):
        test_x = self.vectorizer.transform(test_x)
        return self.model.evaluate(test_x, test_y)

    def get_history(self):
        return self.history

    def plot_model_specific_metrics(self):
        visualizer(self.model, format='png', view=True)
