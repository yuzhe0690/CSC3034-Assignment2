import pandas as pd
import re
import nltk


from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# read and load csv file from file path and return lowercase dataframe
def read_data(file_path):
    df = pd.read_csv(file_path)
    if df is None:
        raise Exception(f"No data found in the file path: {file_path}")
    else:
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



# stemming text
def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]
    


# data preprocessing
def data_preprocessing(data):
    print("\n> Starting data pre-processing...")
    df = data.copy() # duplicate data
    
    print("\t> Removing duplicated datas...")
    df = df.drop_duplicates(subset=['review'], keep='first') # remove duplicated reviews
    print("\t> Removing HTML tags...")
    df['review'] = df['review'].apply(remove_tags) # remove html tags
    print("\t> Removing URLs...")
    df['review'] = df['review'].apply(remove_urls) # remove urls
    print("\t> Removing special characters...")
    df['review'] = df['review'].apply(remove_special_char) # remove special characters
    print("\t> Removing non-word and non-whitespace characters...")
    df = df.replace(to_replace=r'^\s*$', value='', regex=True) # remove non-word and non-whitespace
    print("\t> Removing digits...")
    df = df.replace(to_replace=r'\d+', value='', regex=True) # remove digits
    print("\t> Tokenizing text...")
    df['review'] = df['review'].apply(word_tokenize) # tokenize text
    print("\t> Stemming words in tokens...")
    df['review'] = df['review'].apply(stem_tokens) # lemmatize tokenized text
    
    return df


if __name__ == "__main__":
    # Initialization
    data = read_data("IMDB Dataset.csv")
    nltk.download('punkt') # download punkt tokenizer
    
    # Data inpsection
    print("\n= Data Inspection: =")
    print(data.isnull().sum())
    print(data[data['review'].duplicated()])
    
    # Data Preprocessing
    data = data_preprocessing(data)
    
    

    
