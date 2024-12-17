import pandas as pd
import re
import nltk

from bs4 import BeautifulSoup
from keras.src.layers import Bidirectional
from nltk.tokenize import word_tokenize
from nltk.stem import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np

# read and load csv file from file path and return lowercase dataframe
def read_data(file_path):
    df = pd.read_csv(file_path, names=["review", "rating"])
    print(df)
    if df is not None:
        print(f"> Loaded {len(df)} rows from {file_path}")
        df = df.map(lambda x: x.lower() if isinstance(x, str) else x)  # Lowercase all strings
        return df


# remove urls from text
def remove_urls(text):
    url_pattern = re.compile(r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?")
    return url_pattern.sub('', text)


# remove html tags from text
def remove_tags(text):
    return BeautifulSoup(text, 'html.parser').get_text()


# remove special characters from text
def remove_special_char(text):
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


# data preprocessing
def data_preprocessing(data):
    print("\n> Starting data pre-processing...")
    df = data.copy()  # duplicate data

    # TODO: 50:50 split for positive and negative reviews

    print("\t> Removing duplicated datas...")
    df = df.drop_duplicates(subset=['Review'], keep='first')  # remove duplicated reviews
    print("\t> Removing non-word and non-whitespace characters...")
    df = df.replace(to_replace=r'^\s*$', value='', regex=True)  # remove non-word and non-whitespace
    print("\t> Removing digits...")
    df["Review"] = df["Review"].replace(to_replace=r'\d+', value='', regex=True)  # remove digits
    print("\t> Removing special characters...")
    df['Review'] = df['Review'].apply(remove_special_char)  # remove special characters
    print("\t> Tokenizing text...")
    df['Review'] = df['Review'].apply(word_tokenize)  # tokenize text
    print("\t> Stemming words in tokens...\n")
    df['Review'] = df['Review'].apply(stem_tokens)

    print(df)
    
    return df


if __name__ == "__main__":
    # Initialization
    df = pd.read_csv("DatasEt.csv")

    lemmatizer = WordNetLemmatizer()
    nltk.download('punkt')  # download punkt
    # print(df)
    # Data inpsection
    print("\nData Inspection")
    print(df.isnull().sum())
    print(df[df['Review'].duplicated()])

    # Data Preprocessing
    data = data_preprocessing(df)
    # print(data)
    length = []
    for string in data["Review"]:
        length.append(len(string))
    print(max(length))
    #data = data[data["Rating"] != 3]
    X = data["Review"].to_numpy()
    y = data["Rating"]
    data["Rating"] = data["Rating"].map({1: 0, 2: 1})
    y = data["Rating"].to_numpy()
    # print(y.value_counts())

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train)

    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)

    x_train = pad_sequences(x_train, maxlen=100, padding='post', truncating='post')
    x_test = pad_sequences(x_test, maxlen=100, padding='post', truncating='post')

    model = Sequential([
        Embedding(input_dim=5000, output_dim=100, input_length=100),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(64,activation="relu"),
        Dense(1, activation='sigmoid')
    ])


    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    print(model.summary())
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=20,
        batch_size=128,
        verbose=2
    )
    print(model.summary())
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=10)
    print(test_accuracy)

    import matplotlib.pyplot as plt

    # Plot accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()




