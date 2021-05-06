"""Module for preprocessing the IMDB data set."""
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import nltk
import unicodedata
import re
import string
import contractions
from sklearn.model_selection import train_test_split

class Preprocessor:
    """Class responsible for preprocessing textual and target data"""
    def remove_html(self, text):
        bs = BeautifulSoup(text, 'html.parser')
        return bs.get_text()

    def remove_accented_chars(self, text):
        new_text = unicodedata.normalize('NFKD', text).encode('ascii',
                                                              'ignore').decode(
                                                                  'utf-8',
                                                                  'ignore')
        return new_text

    def remove_stopwords(self, text):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        return ' '.join(filtered_sentence)

    def remove_custom_stopwords(self, text):
        stop_words = ['movie', 'film']
        word_tokens = word_tokenize(text)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        return ' '.join(filtered_sentence)


    def remove_urls(self, text):
        pattern_url = re.compile(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', re.I)
        text = pattern_url.sub('URL', text)
        return text

    def remove_special_chars(self, text):
        pattern_special = re.compile(r'[^a-zA-z0-9.,!?/:;\"\'\s]')
        text = pattern_special.sub('', text)
        return text

    def remove_punctuation(self, text):
        return ''.join([c for c in text if c not in string.punctuation])

    def remove_numbers(self, text):
        return ''.join([c for c in text if not c.isdigit()])

    def apply_stemming(self, text):
        lemmatizer = SnowballStemmer('english')
        new_words = []
        for w in word_tokenize(text):
            new_words.append(lemmatizer.stem(w))

        return ' '.join(new_words)

    def lemmatize(self, text):
        lemma = nltk.wordnet.WordNetLemmatizer()
        new_words = []
        for w in word_tokenize(text):
            new_words.append(lemma.lemmatize(w, pos='v'))
        return ' '.join(new_words)

    def remove_spaces(self, text):
        return ' '.join(text.split())


    def preprocess_text(self, df, text_column_name):
        df[text_column_name] = df[text_column_name].astype(str)
        df[text_column_name] = df[text_column_name].str.lower()
        print('Removing Accented Charachters/URLS/Numbers/Special Charachters')
        df[text_column_name] = [self.remove_accented_chars(r)
                                for r in df[text_column_name]]
        df[text_column_name] = [self.remove_html(r)
                                for r in df[text_column_name]]
        df[text_column_name] = [self.remove_urls(r)
                                for r in df[text_column_name]]
        df[text_column_name] = [self.remove_special_chars(r)
                                for r in df[text_column_name]]
        df[text_column_name] = [self.remove_numbers(r)
                                for r in df[text_column_name]]
        print('Fixing Contractions')
        df[text_column_name] = [contractions.fix(r)
                                for r in df[text_column_name]]
        print('Removing punctuations')
        df[text_column_name] = [self.remove_punctuation(r)
                                for r in df[text_column_name]]
        print('Removing extra spaces')
        df[text_column_name] = [self.remove_spaces(r)
                                for r in df[text_column_name]]
        print('Lematizing')
        df[text_column_name] = [self.lemmatize(r)
                                for r in df[text_column_name]]
        print('Removing custom stop words')
        df[text_column_name] = [self.remove_custom_stopwords(r)
                                for r in df[text_column_name]]
        print('Removing stop words')
        df[text_column_name] = [self.remove_stopwords(r)
                                for r in df[text_column_name]]
        return df

    def preprocess_target(self, df, target_label_name):
        df.loc[df[target_label_name] == 'positive', target_label_name] = int(0)
        df.loc[df[target_label_name] == 'negative', target_label_name] = int(1)
        return df

    def preprocess_df(self, df, text_column_name, target_label_name):
        df = self.preprocess_text(df, text_column_name)
        print('Succesfully preprocessed the descriptive features')

        return self.preprocess_target(df, target_label_name)

    def split(self, df, text_column_name, target_label_name):
        train_x, test_x, train_y, test_y = train_test_split(
            df[text_column_name],
            df[target_label_name],
            test_size=0.2,
            random_state=42)

        train_x, val_x, train_y, val_y = train_test_split(train_x,
                                                          train_y,
                                                          test_size=0.2,
                                                          shuffle=True)
        train_y = np.array(train_y).astype('float32')
        test_y = np.array(test_y).astype('float32')
        val_y = np.array(val_y).astype('float32')

        return train_x, test_x, val_x, train_y, test_y, val_y
