"""Module containing evaulation metrics for models."""
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
class Evaluator:
    '''
    Evaulates and compares models.'''
    def __init__(self, test_x, test_y):
        self.test_x = test_x
        self.test_y = test_y
        self.predictions = {}
        self.accuracies = {}
        self.losses = {}

    def evaluate_models(self, models):
        for model in models:
            print('----------------------------------------------------------')
            print(f'---Evaluating {model.get_name()}---')
            score = model.evaulate(self.test_x, self.test_y)
            prediction = model.get_prediction(self.test_x)
            self.accuracies[model.get_name()] = score[1]
            self.losses[model.get_name()] = score[0]
            self.plot(model)
            self.plot_confusion_mtx(prediction)

        self.plot_comparision()

    def evaluate(self, preds):
        pass

    def plot(self, model):
        self.plot_accuracy(model)
        self.plot_losses(model)

    def plot_accuracy(self, model):
        history = model.get_history().history

        train_acc = history['binary_accuracy']
        val_acc = history['val_binary_accuracy']
        epochs = range(1, len(val_acc) + 1)

        plt.plot(epochs, train_acc, label='Training Accuracy', color='#377eb8')
        plt.plot(epochs, val_acc, label='Validation Accuracy', color='#a65628')
        plt.title('{0}: Training and validation Accuracy'.format(
            model.get_name()))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.show()

    def plot_losses(self, model):
        history = model.get_history().history

        train_loss = history['loss']
        val_loss = history['val_loss']
        epochs = range(1, len(val_loss) + 1)

        plt.clf()
        plt.plot(epochs, train_loss, label='Training loss', color='#377eb8')
        plt.plot(epochs, val_loss, label='Validation Loss', color='#a65628')
        plt.title('{0}: Training and validation loss'.format(
            model.get_name()))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.show()

    def plot_comparision_acc(self):
        plt.bar(self.accuracies.keys(), self.accuracies.values())
        plt.title('Accuracy of Models')
        plt.ylabel('Accuracy')
        plt.xlabel('Model Name')
        plt.show()

    def plot_comparision_loss(self):
        plt.bar(self.losses.keys(), self.losses.values())
        plt.title('Losses of Models')
        plt.ylabel('Loss')
        plt.xlabel('Model Name')
        plt.show()

    def plot_comparision(self):
        self.plot_comparision_acc()
        self.plot_comparision_loss()

    def plot_confusion_mtx(self, prediction):
        print('Confusion Matrix:')
        print(confusion_matrix(self.test_y, prediction))

        print('\n')
        print('Classification Report')
        print(classification_report(self.test_y, prediction))
        print('\n')
