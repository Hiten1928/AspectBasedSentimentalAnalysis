import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

ef create_sentiments_pandas_frame():
    data = pd.read_csv("C:\\Users\ROSS\Documents\Study\Software Quality\Train_Set.csv")
    processed_data = data.copy()
    processed_data['sentiment'] = processed_data['Polarity'].apply(lambda x: 0 if x == 'negative' else 1)
    return processed_data
