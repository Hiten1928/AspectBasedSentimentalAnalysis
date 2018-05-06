import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# Function to read csv & create pandas dataframe
def create_sentiments_pandas_frame():
    data = pd.read_csv("C:\\Users\ROSS\Documents\Study\Software Quality\Train_Set.csv")
    processed_data = data.copy()
    processed_data['sentiment'] = processed_data['Polarity'].apply(lambda x: 0 if x == 'negative' else 1)
    processed_data['text_clean'] = processed_data['Text']
    processed_data = processed_data.loc[:, ['text_clean', 'sentiment']]
    return processed_data


# Function to vectorize and split the data set
def create_sentiments_test_train_set(processed_data):
    X = processed_data['text_clean']
    y = processed_data['sentiment']

    # Using CountVectorizer to convert text into tokens/features
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 1), max_df=.80, min_df=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)
    # Using training data to transform text into counts of features for each message
    vectorizer.fit(X_train)
    X_train_dtm = vectorizer.transform(X_train)
    X_test_dtm = vectorizer.transform(X_test)

    return X, y, X_train, X_test, y_train, y_test, X_train_dtm, X_test_dtm, vectorizer
