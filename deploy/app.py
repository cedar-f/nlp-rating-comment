from types import MethodType
from flask.globals import request
import numpy as np
from flask import Flask, Request, jsonify, render_template
import pickle as pkl
import keras


import keras
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
import string
from underthesea import word_tokenize
# from gensim.utils import simple_preprocess
# from gensim.models.wrappers import FastText
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers import Dense, Lambda, dot, Activation, concatenate, Embedding, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization, AveragePooling1D
# import gensim as gs
import tensorflow as tf

# from gensim.models import FastText
sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
app = Flask(__name__)


def word_processing_teacher(sentence, stopwords):
    sentence = " ".join(simple_preprocess(sentence))
    sentence = [word for word in word_tokenize(
        sentence.lower(), format="text").split() if word not in stopwords]
    return [word for word in sentence if word != ""]
# input_sentences = [word_processing_teacher(str(sentence),stopwords) for sentence in dataset["content"].values.tolist()]


def clean_text(text):
    # preprocessing ....
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.translate(str.maketrans('', '', string.digits))
    # Remove puncuation
    text = text.translate(string.punctuation)
    # Convert words to lower case and split them
    text = text.lower().split()
    text = " ".join(text)
    return text


def create_stopwords(path, original_stopwords):
    with open(path, encoding="utf-8") as words:
        return [w[:len(w) - 1] for w in words] + original_stopwords


original_stopwords = ["tiki", "lazada", "shopee"]
stopwords = create_stopwords(
    './input/stopword/vietnamese-stopwords-dash.txt', original_stopwords)
word2id__ = np.load(
    './input/dict/word_2_dict_id.npy', allow_pickle='TRUE').item()
pathmodel = "./model/PHU_BI_LSTM_CNN.06.hdf5"
model = keras.models.load_model(pathmodel)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    content_predict = request.form.get('content_predict')
    if(content_predict != ""):
        try:
            cleanText = clean_text(content_predict)
            tokenized_s = word_processing_teacher(cleanText, stopwords)
            encoded_s = [[word2id__[word.lower()] for word in tokenized_s]]
            encoded_s = pad_sequences(encoded_s, maxlen=200)
            label = model.predict(np.array(encoded_s))
            # label
            # result = 'content_predict ' + \
            #     format(content_predict) + ' => ' + \
            #     'result_predict ' + format(np.argmax(label))
            result = "has not been predicted"
            if(np.argmax(label) == 0):
                result = "rất không hài lòng"

            if(np.argmax(label) == 1):
                result = "không hài lòng"

            if(np.argmax(label) == 2):
                result = "phân vân"

            if(np.argmax(label) == 3):
                result = "hài lòng"

            return render_template('index.html', content=content_predict, outcome=result)
        except:
            return render_template('index.html', content="Not predictable due to limited dictionary", outcome="no result")
    else:
        return render_template('index.html', content="No data available", outcome="No data available")

    # return render_template('index.html', outcome=content_predict)
    # pass


if __name__ == "__main__":
    app.run(debug=True, port=5001)
