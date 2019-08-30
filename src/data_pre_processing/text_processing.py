import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

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



def preprocess_text(csv_file, processed_file):
    "Function to prepare train test dataset"

    data_frame = pd.read_csv(csv_file, usecols=["clean_text","emotion"])

    data_frame['clean_text'] = data_frame['clean_text'].fillna("NONE")

    #Removing Stop words from the clean text
    data_frame['clean_text'] = data_frame['clean_text'].apply(lambda x : remove_stop_words(x))
    data_frame['clean_text'] = data_frame['clean_text'].fillna("NONE")

    # Applying Stemmer
    data_frame['clean_text'] = data_frame['clean_text'].apply(lambda x : get_stemmed_text(x))
    data_frame['clean_text']  = data_frame['clean_text'].fillna("NONE")

    # Applying Lemmatizer
    data_frame['clean_text'] = data_frame['clean_text'].apply(lambda x : get_lemmatized_text(x))

    print(data_frame.head())
    data_frame.to_csv(processed_file)

def prepare_train_test_data(processed_file):
    "Function to vectorize input data"
    data_frame = pd.read_csv(processed_file)
    print("data_frame : ",data_frame.head())
    targets = data_frame['emotion'].values
    tweets = data_frame['clean_text'].values
    print(targets[:10])
    print(tweets[:10])
    # # Create feature vectors
    # vectorizer = TfidfVectorizer(min_df=5,
    #                              max_df=0.8,
    #                              sublinear_tf=True,
    #                              use_idf=True)
    
    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3))


if __name__ == '__main__':
    print()
    csv_file = "C:\\Users\\Anurag\\PycharmProjects\\DeepLearning\\emotion-analysis\\dataset\\cleaned_tweets.csv"
    processed_file = "C:\\Users\\Anurag\\PycharmProjects\\DeepLearning\\emotion-analysis\\dataset\\processed_tweets.csv"
    # preprocess_text(csv_file=csv_file,processed_file=processed_file)
    prepare_train_test_data(processed_file=processed_file)
