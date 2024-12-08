import pandas as pd
import re
import nltk


from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


# read and load csv file from file path and return lowercase dataframe
def read_data(file_path):
    df = pd.read_csv(file_path)
    if df is not None:
        print(f"> Loaded {len(df)} rows from {file_path}")
        df = df.map(lambda x: x.lower() if isinstance(x, str) else x) # Lowercase all strings
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


# lemmatize text
def lemmatize_tokens(tokens):
    # convert POS tag to WordNet format
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    # lemmatize tokens
    lemmas = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
    return lemmas
    


# data preprocessing
def data_preprocessing(data):
    print("> Starting data pre-processing...")
    df = data.copy() # duplicate data
    stop_words = set(stopwords.words('english')) # set of stopwords
    
    # TODO: 50:50 split for positive and negative reviews
    
    print("\t> Removing duplicated datas...")
    df = df.drop_duplicates(subset=['review'], keep='first') # remove duplicated reviews
    print("\t> Removing non-word and non-whitespace characters...")
    df = df.replace(to_replace=r'^\s*$', value='', regex=True) # remove non-word and non-whitespace
    print("\t> Removing digits...")
    df = df.replace(to_replace=r'\d+', value='', regex=True) # remove digits
    print("\t> Removing URLs...")
    df['review'] = df['review'].apply(remove_urls) # remove urls
    print("\t> Removing HTML tags...")
    df['review'] = df['review'].apply(remove_tags) # remove html tags
    print("\t> Removing special characters...")
    df['review'] = df['review'].apply(remove_special_char) # remove special characters
    print("\t> Tokenizing text...")
    df['review'] = df['review'].apply(word_tokenize) # tokenize text
    print("\t> Removing stopwords in tokens...")
    df['review'] = df['review'].apply(lambda x: [word for word in x if word not in stop_words]) # remove stopwords
    print("\t> Lemmatizing text in tokens...")
    df['review'] = df['review'].apply(lemmatize_tokens) # lemmatize tokenized text
    
    return df


if __name__ == "__main__":
    # Initialization
    data = read_data("IMDB Dataset.csv")
    lemmatizer = WordNetLemmatizer()
    nltk.download('punkt') # download punkt
    nltk.download('stopwords') # download stopwords
    nltk.download('wordnet') # download wordnet
    nltk.download('omw-1.4') # download wordnet
    nltk.download('averaged_perceptron_tagger')
    
    # Data inpsection
    print("\nData Inspection")
    print(data.isnull().sum())
    print(data[data['review'].duplicated()])
    
    # Data Preprocessing
    data = data_preprocessing(data)
    
    

    
