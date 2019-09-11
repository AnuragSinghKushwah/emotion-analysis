import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
from nltk.tokenize import WordPunctTokenizer

def clean_csv(file_name, output_csv):
    "Function to input csv file"

    # Reading CSV File
    data_frame = pd.read_csv(file_name)
    print("------------------------------------")

    # Finding Emotions Value Counts
    print(data_frame.emotion.value_counts())

    # Droping ID Column
    data_frame.drop(['tweet_id'], axis=1, inplace=True)

    # Finding Length of Tweets before Cleaning
    data_frame['pre_clean_len'] = [len(t) for t in data_frame['tweet'].values]

    # Removing HTML Tags from the tweets
    data_frame["clean_text"] = data_frame['tweet'].apply(lambda x : remove_html(input_str=x))

    #Removing User Mentions from the Tweets
    data_frame["clean_text"] = data_frame['clean_text'].apply(lambda x : remove_mentions(input_str=x))

    #Removing URL Tags from the tweets
    data_frame["clean_text"] = data_frame['clean_text'].apply(lambda x : remove_urls(input_str=x))

    #Removing Special Characters and Punctuations
    data_frame["clean_text"] = data_frame['clean_text'].apply(lambda x : remove_spl_characters(input_str=x))

    # Removing Unicodes from the Text
    data_frame["clean_text"] = data_frame['clean_text'].apply(lambda x : remove_unicodes(input_str=x))

    #Changing Chase
    data_frame["clean_text"] = data_frame['clean_text'].apply(lambda x : x.lower())

    # Removing Extra Spaces and Tokenizing Text
    data_frame["clean_text"] = data_frame['clean_text'].apply(lambda x : tokenize_string(input_str=x))

    # Filling Null Values after cleaning of text
    data_frame['clean_text'] = data_frame['clean_text'].fillna("NULL")

    # Finding length of text after cleaning
    data_frame['post_clean_len'] = [len(t) for t in data_frame['clean_text'].values]

    print(data_frame.head())
    data_frame.to_csv(output_csv)

def remove_html(input_str):
    "Function to remove html from the input string"
    return BeautifulSoup(input_str, 'lxml').get_text()

def remove_mentions(input_str):
    "Function to remove user mentions from the tweet"
    return re.sub(r'@[A-Za-z0-9_-]+', '', input_str)

def remove_urls(input_str):
    "Function to remove URLS from the tweet"
    return re.sub('https?://[A-Za-z0-9./]+','',input_str)

def remove_spl_characters(input_str):
    "Function to remove Special Characters from the Tweet"
    return  re.sub("[^a-zA-Z]", " ", input_str)

def remove_unicodes(input_str):
    "Function to remove Unicodes from the String"
    try:
        clean = input_str.decode("utf-8-sig").replace(u"\ufffd", "?")
        return clean
    except Exception as e:
        print("exception in removing unicodes : ",e)
        return input_str

def tokenize_string(input_str):
    "Function to remove extra spaces and create tokens"
    tok = WordPunctTokenizer()
    try:
        words = tok.tokenize(input_str)
        return " ".join(words).strip()
    except Exception as e:
        print('Exception in tokenizing string : ',e)
        return input_str

if __name__ == '__main__':
    file_name = "C:\\Users\\Anurag\\PycharmProjects\\DeepLearning\\emotion-analysis\\dataset\\smile-annotations-final.csv"
    output_csv = "C:\\Users\\Anurag\\PycharmProjects\\DeepLearning\\emotion-analysis\\dataset\\cleaned_tweets_1.csv"
    clean_csv(file_name=file_name, output_csv=output_csv)


