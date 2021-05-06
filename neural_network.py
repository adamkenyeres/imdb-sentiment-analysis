"""Module for neural network model."""
from model import Model
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras import losses

class NeuralNetwork(Model):
    """Base class for neural network."""

    def __init__(self, name, vectorizer, features = 10000, epochs = 10,
                 use_embeding=True):
        Model.__init__(self, name, vectorizer)
        self.features = features
        self.epochs = epochs
        self.model = None
        self.history = None
        self.use_embeding = use_embeding


    def build(self):
        self.model = Sequential()
        if self.use_embeding:
            print('Using embedding')
            self.model.add(layers.Embedding(self.features + 1, 16))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.GlobalAveragePooling1D())
            self.model.add(layers.Dropout(0.2))
        else:
            print('Not using')
            self.model.add(layers.Dense(units=128,input_dim=self.features,
                                        activation='relu'))

        self.model.add(layers.Dense(units=128, activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(units=128, activation='relu'))
        self.model.add(layers.Dense(units=64, activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(units=32, activation='relu'))
        self.model.add(layers.Dense(1))

        self.model.summary()
        self.model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                optimizer='adam',
                metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

    def train(self, train_X, train_Y, val_X, val_Y):
        train_X = self.vectorizer.fit_transform(train_X)
        val_X = self.vectorizer.transform(val_X)

        print(train_X.shape)
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
        pass
