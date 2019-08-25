import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

file_name = "C:\\Users\\Anurag\\PycharmProjects\\DeepLearning\\emotion-analysis\\dataset\\smile-annotations-final.csv"
data_frame = pd.read_csv(file_name)
print(data_frame.head())
print("------------------------------------")
print(data_frame.label.value_counts())