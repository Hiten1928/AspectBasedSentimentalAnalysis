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


# Function to implement prediction using naive bayesian
def prediction_naive_bayesian(X_train_dtm, y_train, X_test_dtm, y_test):
    NB = MultinomialNB()
    NB.fit(X_train_dtm, y_train)
    y_pred = NB.predict(X_test_dtm)
    print_metric_results("Naive Bayes", y_test, y_pred)


#Function to implement prediction using logistic regression
def prediction_logistic_regression(X_train_dtm, y_train,X_test_dtm, y_test):
    LR = LogisticRegression()
    LR.fit(X_train_dtm, y_train)
    y_pred = LR.predict(X_test_dtm)
    print_metric_results("Logistic Regression", y_test, y_pred)


#Function to implement prediction using linear svm
def prediction_linear_svm(X_train_dtm, y_train, X_test_dtm, y_test):
    SVM = LinearSVC()
    SVM.fit(X_train_dtm, y_train)
    y_pred = SVM.predict(X_test_dtm)
    print_metric_results("SVM", y_test, y_pred)


#Function to implement prediction using KNN
def predict_knn(X_train_dtm, y_train, X_test_dtm, y_test):
    KNN = KNeighborsClassifier(n_neighbors=3)
    KNN.fit(X_train_dtm, y_train)
    y_pred = KNN.predict(X_test_dtm)
    print_metric_results("K Nearest Neighbors (NN = 3)", y_pred, y_test)


#Function to print_metric_prediction_results
def print_metric_results(classifier,y_test, y_pred):
    print(classifier)
    print('Accuracy Score: ', metrics.accuracy_score(y_test, y_pred) * 100, '%', sep='')
    print('Precision Score: ', metrics.precision_score(y_test, y_pred) * 100, '%', sep='')
    print('Recall Score: ', metrics.recall_score(y_test, y_pred) * 100, '%', sep='')
    print('F1-Score: ', metrics.f1_score(y_test, y_pred) * 100, '%', sep='')
    print('Confusion Matrix: ', metrics.confusion_matrix(y_test, y_pred), sep='\n')


#Function to print tokens and their sentiment type from the train set
def analyze_custom_input_review_sentiment(test, X, y):
    trainingVector = CountVectorizer(stop_words='english', ngram_range=(1, 1), max_df=.80, min_df=5)
    trainingVector.fit(X)
    X_dtm = trainingVector.transform(X)
    nb_complete = MultinomialNB()
    nb_complete.fit(X_dtm, y)

    test_dtm = trainingVector.transform(test)
    predicted_label = nb_complete.predict(test_dtm)
    tags = ['Negative', 'Positive']
    print('The review sentiment is ', tags[predicted_label[0]])


#Function to print sentiment predictions for custom review input by user
def print_review_sentiment(vectorizer, X_train_dtm, X_test_dtm, y_train, X_test, y_test):
    NB = MultinomialNB()
    NB.fit(X_train_dtm, y_train)
    tokens_words = vectorizer.get_feature_names()
    print('\nAnalysis')
    print('No. of tokens: ', len(tokens_words))
    counts = NB.feature_count_
    df_table = {'Token': tokens_words, 'Negative': counts[0, :], 'Positive': counts[1, :]}

    tokens = pd.DataFrame(df_table, columns=['Token', 'Positive', 'Negative'])
    positives = len(tokens[tokens['Positive'] > tokens['Negative']])
    print("token_words", tokens_words)
    print('No. of positive tokens: ', positives)
    print('No. of negative tokens: ', len(tokens_words) - positives)
    print("**************************************************************************************************************")
    print("Identified Positive/Negative tokens and their values")
    print("tokens", tokens)


def main():
    processed_data = create_sentiments_pandas_frame()
    X, y, X_train, X_test, y_train, y_test, X_train_dtm, X_test_dtm, vectorizer = create_sentiments_test_train_set(processed_data)
    prediction_naive_bayesian(X_train_dtm, y_train, X_test_dtm, y_test)
    prediction_logistic_regression(X_train_dtm, y_train, X_test_dtm, y_test)
    prediction_linear_svm(X_train_dtm, y_train, X_test_dtm, y_test)
    predict_knn(X_train_dtm, y_train, X_test_dtm, y_test)
    print_review_sentiment(vectorizer, X_train_dtm, X_test_dtm, y_train, X_test, y_test)

    custom_review = []
    print("**************************************************************************************************************")
    print('Enter review to predict sentiment: ', end=" ")
    custom_review.append(input())
    analyze_custom_input_review_sentiment(custom_review, X, y)


if __name__ == '__main__':
    main()
