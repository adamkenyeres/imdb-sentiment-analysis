"""Module containing methods for data understanding."""

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import  CountVectorizer
from yellowbrick.text import FreqDistVisualizer

class DataUnderstanding:
    '''
    Resplonsible for plotting graphs of the data.'''
    def plot_target_distribution(self, df):
        names = ['Positive', 'Negative']
        sizes = [len(df[df['sentiment'] == 'positive']),
                 len(df[df['sentiment'] == 'negative'])]

        plt.bar(names, sizes, color ='maroon', width = 0.4)

        plt.ylabel('Count')
        plt.title('Distribution of the sentiments')
        plt.show()

    def plot_most_frequent_tokens(self, df, column_name):
        count_vectorizer = CountVectorizer()
        tf_original = count_vectorizer.fit_transform(df[column_name])
        tf_feature_names = count_vectorizer.get_feature_names()
        visualizer = FreqDistVisualizer(features=tf_feature_names, orient='v')
        visualizer.fit(tf_original)
        visualizer.show()
