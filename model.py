'''
Module for all models'''
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard

class Model:
    '''
    Base class for all models'''
    def __init__(self, name, vectorizer):
        self.name = name
        self.vectorizer = vectorizer
        self.tensorboard = TensorBoard(log_dir='logs1_/{}'.format(name))

    def get_name(self):
        return self.name

    def train(self, train_x, train_y):
        pass

    def plot_model_specific_metrics(self):
        pass

    def plot_architecture(self):
        plot_model(self.model, to_file='{0}.png'.format(self.name))

    def get_prediction(self, review):
        reviews = self.vectorizer.transform(review)
        return self.model.predict_classes(reviews)
