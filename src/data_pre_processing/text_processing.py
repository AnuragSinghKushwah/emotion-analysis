import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

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

    data_frame = data_frame[data_frame['clean_text'].notnull()]
    print("data_frame : ",len(data_frame))

    # data_frame['clean_text'] = data_frame['clean_text'].fillna("NONE")
    #Removing Stop words from the clean text
    data_frame['clean_text'] = data_frame['clean_text'].apply(lambda x : remove_stop_words(x))
    # data_frame['clean_text'] = data_frame['clean_text'].fillna("NONE")

    # Applying Stemmer
    data_frame['clean_text'] = data_frame['clean_text'].apply(lambda x : get_stemmed_text(x))
    # data_frame['clean_text']  = data_frame['clean_text'].fillna("NONE")

    # Applying Lemmatizer
    data_frame['clean_text'] = data_frame['clean_text'].apply(lambda x : get_lemmatized_text(x))
    # data_frame['clean_text'] = data_frame['clean_text'].fillna("NONE")

    # Filtering none values
    filtered_df = data_frame[data_frame['clean_text'].notnull()]
    # filtered_df = filtered_df[filtered_df['clean_text']!="NONE"]

    print("filtered_df : ",len(filtered_df))
    print(filtered_df.head())
    data_frame.to_csv(processed_file)

def prepare_train_test_data(processed_file):
    "Function to vectorize input data"
    data_frame = pd.read_csv(processed_file)
    print("data_frame : ",data_frame.head())
    data_frame = data_frame[data_frame['clean_text'].notnull() | data_frame['clean_text'].notna()] #df[df['name'].notnull() | df['foo'].notnull()]
    targets = data_frame['emotion'].values
    tweets = data_frame['clean_text'].values
    print("total_tweets : ",len(targets))
    print("total_targets: ",len(tweets))

    tweets_train_clean = tweets[:round(len(tweets)*0.7)]
    targets_train_clean = targets[:round(len(targets)*0.7)]
    tweets_test_clean = tweets[round(len(tweets)*0.7):]
    targets_test_clean = targets[round(len(targets)*0.7):]

    print("tweets_train_clean  : ",len(tweets_train_clean))
    print("targets_train_clean : ",len(targets_train_clean))
    print("tweets_test_clean   : ",len(tweets_test_clean))
    print("targets_test_clean  : ",len(targets_test_clean))

    # # Create feature vectors
    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3))
    ngram_vectorizer.fit(tweets_train_clean)
    X = ngram_vectorizer.transform(tweets_train_clean)
    X_test = ngram_vectorizer.transform(tweets_test_clean)

    X_train, X_val, y_train, y_val = train_test_split(X, targets_train_clean, train_size=0.75)

    for c in [0.01, 0.05, 0.25, 0.5, 1]:
        lr = LogisticRegression(C=c)
        lr.fit(X_train, y_train)
        print("Logistic Regression : Accuracy for C=%s: %s"% (c, accuracy_score(y_val, lr.predict(X_val))))

    final_ngram = LogisticRegression(C=0.5)
    final_ngram.fit(X, targets_train_clean)
    print("Logistic Regression : Final Accuracy: %s"
          % accuracy_score(targets_test_clean, final_ngram.predict(X_test)))

    print('---------------------------------------------------------')

    for c in [0.01, 0.05, 0.25, 0.5, 1]:
        svm = LinearSVC(C=c)
        svm.fit(X_train, y_train)
        print("SVC : Accuracy for C=%s: %s"% (c, accuracy_score(y_val, svm.predict(X_val))))

    final_svm_ngram = LinearSVC(C=0.01)
    final_svm_ngram.fit(X, targets_train_clean)
    print("SVC : Final Accuracy: %s"% accuracy_score(targets_test_clean, final_svm_ngram.predict(X_test)))


if __name__ == '__main__':
    print()
    csv_file = "C:\\Users\\Anurag\\PycharmProjects\\DeepLearning\\emotion-analysis\\dataset\\cleaned_tweets.csv"
    processed_file = "C:\\Users\\Anurag\\PycharmProjects\\DeepLearning\\emotion-analysis\\dataset\\processed_tweets.csv"
    # preprocess_text(csv_file=csv_file,processed_file=processed_file)
    prepare_train_test_data(processed_file=processed_file)
#