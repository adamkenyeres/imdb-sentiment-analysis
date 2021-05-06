"""Entry point for model training."""
import pandas as pd
from vectorizer import CountVectorization, TextVectorizer
from preprocessor import Preprocessor
from data_understanding import DataUnderstanding
from neural_network import NeuralNetwork
from lstm import LSTM
from evaluator import Evaluator

DATAUNDERSTANDING_ENABLED = False

def preprocess_and_split(data_frame, du, pr):
    if DATAUNDERSTANDING_ENABLED:
        du.plot_target_distribution(data_frame)
        du.plot_most_frequent_tokens(df, 'review')

    data_frame = pr.preprocess_df(data_frame, 'review', 'sentiment')

    if DATAUNDERSTANDING_ENABLED:
        du.plot_most_frequent_tokens(data_frame, 'review')

    return pr.split(data_frame, 'review', 'sentiment')


models = []
data_understander = DataUnderstanding()
text_vectorizer = TextVectorizer('Text Vectorizer', 1000, 250)
count_vectorizer = CountVectorization('Count Vectorizer', 1000)
preprocessor = Preprocessor()

df = pd.read_csv('IMDB Dataset.csv')
df.head()
train_x, test_x, val_x, train_y, test_y, val_y = preprocess_and_split(
    df,
    data_understander,
    preprocessor)

evaulator = Evaluator(test_x, test_y)
train_x.head()

text_vectorized_nn = NeuralNetwork('Text Vectorized NN Model',
                                   text_vectorizer,
                                   1000)
text_vectorized_nn.build()
text_vectorized_nn.plot_architecture()
text_vectorized_nn.train(train_x, train_y, val_x, val_y)
models.append(text_vectorized_nn)

text_vectorized_lstm = LSTM('LSTM', text_vectorizer)
text_vectorized_lstm.build()
text_vectorized_lstm.plot_architecture()
text_vectorized_lstm.train(train_x, train_y, val_x, val_y)
models.append(text_vectorized_lstm)

count_vectorized_nn = NeuralNetwork('Count Vectorized NN Model',
                              count_vectorizer,
                              1000, use_embeding=False)
count_vectorized_nn.build()
count_vectorized_nn.train(train_x, train_y, val_x, val_y)
models.append(count_vectorized_nn)

evaulator.evaluate_models(models)
