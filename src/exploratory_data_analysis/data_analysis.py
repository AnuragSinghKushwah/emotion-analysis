import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def emotion_counts(data_frame):
    "Function to find Emotions Value Counts"
    emotion_count = data_frame['emotion'].value_counts()
    print(emotion_count.items())
    x_values=[]
    y_values=[]
    for i in emotion_count.items():
        x_values.append(i[0])
        y_values.append(i[1])
    plot_bar_x(label=x_values,data=y_values,x_label="Emotion Type",y_label='Frequency Count',title='Emotion Frequency plot')

def plot_text_lengths(data_frame):
    "Function to plot text lengths on box plot"
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.boxplot([data_frame.pre_clean_len, data_frame.post_clean_len])
    plt.show()

def plot_bar_x(label, data, x_label, y_label, title):
    # this is for plotting purpose
    index = np.arange(len(label))
    my_colors = 'rgbyc'
    plt.bar(index, data,color=my_colors)
    plt.xlabel(x_label, fontsize=5)
    plt.ylabel(y_label, fontsize=5)
    plt.xticks(index, label, fontsize=5, rotation=90)
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    csv_file = "C:\\Users\\Anurag\\PycharmProjects\\DeepLearning\\emotion-analysis\\dataset\\cleaned_tweets.csv"
    data_frame = pd.read_csv(csv_file)

    # Finding Emotion Counts
    emotion_counts(data_frame=data_frame)

    # Plotting Length of text before and after data cleaning
    # plot_text_lengths(data_frame=data_frame)