import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

def remove_stop_words(input_str):
    # Function to remove stop words from input string
    english_stop_words = stopwords.words('english')
    try:
        return " ".join([word for word in input_str.split() if word not in english_stop_words])
    except Exception as e:
        print("Exception in removing stopwords : ",e)
        return input_str

def get_stemmed_text(input_str):
    # Function to stem tokens of input string
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    try:
        return ' '.join([stemmer.stem(word) for word in input_str.split()])
    except Exception as e:
        print("Exception in stemming string : ",e)
        return input_str

def get_lemmatized_text(input_str):
    # Function to lemmatize tokens of input string
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    try:
        return ' '.join([lemmatizer.lemmatize(word) for word in input_str.split()])
    except Exception as e:
        print("Exception in lemmatizing string : ",e)
        return input_str

def vectorize_data():
    "Function to vectorize input data"
    # Create feature vectors
    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df=0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    
if __name__ == '__main__':
    print()