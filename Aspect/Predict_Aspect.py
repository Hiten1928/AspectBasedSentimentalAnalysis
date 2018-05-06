import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


#Function to read csv & create pandas dataframe
def create_aspects_pandas_frame():
    data = pd.read_csv("Game_Review_Train.csv")
    # data = pd.read_csv("C:\\Users\ROSS\Documents\Study\Software Quality\Train_Set.csv")

    processed_data = data.copy()
    processed_data['aspect'] = processed_data['Category']
    processed_data['aspect'] = process_aspect(processed_data)

    processed_data['text_clean'] = processed_data['Text']
    processed_data = processed_data.loc[:, ['text_clean', 'aspect']]
    return processed_data


#Function to convert aspects into numerical form
def process_aspect(data_clean):
    for i in range (len(data_clean['aspect'])):
        if data_clean.loc[i, 'aspect'] == 'combat':
            data_clean.loc[i, 'aspect'] = 0
        if data_clean.loc[i, 'aspect'] == 'gameplay':
            data_clean.loc[i, 'aspect'] = 1
        if data_clean.loc[i, 'aspect'] == 'action':
            data_clean.loc[i, 'aspect'] = 2
    return data_clean['aspect']


#Function to vectorize and split the data set
def create_aspect_test_train_set(processed_data):
    X = processed_data['text_clean']
    y = processed_data['aspect']

    # Using CountVectorizer to convert text into tokens/features
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 1), max_df=.80, min_df=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)
    # Using training data to transform text into counts of features for each message
    vectorizer.fit(X_train)
    X_train_dtm = vectorizer.transform(X_train)
    X_test_dtm = vectorizer.transform(X_test)

    return X, y, X_train, X_test, y_train, y_test, X_train_dtm, X_test_dtm, vectorizer


#Function to print_metric_prediction_results
def print_metric_results(classifier,y_test, y_pred):
    print("\n")
    print(classifier)
    print('Accuracy Score: ', metrics.accuracy_score(y_test, y_pred) * 100, '%', sep='')
    print(metrics.classification_report(y_test, y_pred))
    print('Confusion Matrix: ', metrics.confusion_matrix(y_test, y_pred), sep='\n')


#Function to implement prediction using naive bayesian
def prediction_naive_bayesian(X_train_dtm, y_train, X_test_dtm, y_test):
    NB = MultinomialNB()
    NB.fit(X_train_dtm, y_train)
    y_pred = NB.predict(X_test_dtm)
    print_metric_results("Naive Bayes", y_test, y_pred)
    return y_pred


#Function to implement prediction using logistic regression
def prediction_logistic_regression(X_train_dtm, y_train,X_test_dtm, y_test):
    LR = LogisticRegression()
    LR.fit(X_train_dtm, y_train)
    y_pred = LR.predict(X_test_dtm)
    print_metric_results("Logistic Regression", y_test, y_pred)
    return y_pred


#Function to implement prediction using linear svm
def prediction_linear_svm(X_train_dtm, y_train, X_test_dtm, y_test):
    SVM = LinearSVC()
    SVM.fit(X_train_dtm, y_train)
    y_pred = SVM.predict(X_test_dtm)
    print_metric_results("SVM", y_test, y_pred)
    return y_pred


#Function to implement prediction using KNN
def predict_knn(X_train_dtm, y_train, X_test_dtm, y_test):
    KNN = KNeighborsClassifier(n_neighbors=3)
    KNN.fit(X_train_dtm, y_train)
    y_pred = KNN.predict(X_test_dtm)
    print_metric_results("K Nearest Neighbors (NN = 3)", y_pred, y_test)
    return y_pred


#Function to print aspect predictions for custom review input by user
def print_review_aspect(vectorizer, X_train_dtm, X_test_dtm, y_train, X_test, y_test):
    NB = MultinomialNB()
    NB.fit(X_train_dtm, y_train)
    tokens_words = vectorizer.get_feature_names()
    print('\nAnalysis')
    print('No. of tokens: ', len(tokens_words))
    counts = NB.feature_count_
    df_table = {'Token': tokens_words, 'combat': counts[0, :], 'gameplay': counts[1, :], 'action': counts[2, :]}

    tokens = pd.DataFrame(df_table, columns=['Token', 'combat', 'gameplay', 'action'])
    print("**************************************************************************************************************")
    print("Identified combat/gameplay/action tokens from data set and their values")
    print("tokens", tokens)


#Function to print tokens and their aspect category from the train set
def analyze_custom_input_review_aspect(test, X, y):
    trainingVector = CountVectorizer(stop_words='english', ngram_range=(1, 1), max_df=.80, min_df=5)
    trainingVector.fit(X)
    X_dtm = trainingVector.transform(X)
    nb_complete = MultinomialNB()
    nb_complete.fit(X_dtm, y)

    test_dtm = trainingVector.transform(test)
    predicted_label = nb_complete.predict(test_dtm)
    tags = ['combat', 'gameplay', 'action']
    print('The review aspect is', tags[predicted_label[0]])


def main():
    processed_data = create_aspects_pandas_frame()
    X, y, X_train, X_test, y_train, y_test, X_train_dtm, X_test_dtm, vectorizer = create_aspect_test_train_set(processed_data)
    y_pred_naive = prediction_naive_bayesian(X_train_dtm, y_train, X_test_dtm, y_test)
    y_pred_logistics = prediction_logistic_regression(X_train_dtm, y_train, X_test_dtm, y_test)
    y_pred_linear_svm = prediction_linear_svm(X_train_dtm, y_train, X_test_dtm, y_test)
    y_pred_knn = predict_knn(X_train_dtm, y_train, X_test_dtm, y_test)
    pred_list = [y_pred_naive, y_pred_logistics, y_pred_linear_svm, y_pred_knn]
    print_review_aspect(vectorizer, X_train_dtm, X_test_dtm, y_train, X_test, y_test)

    # custom_review = []
    # print(
    #     "*************************************************************************************************************")
    # print('Enter review to predict aspect: ', end=" ")
    # custom_review.append(input())
    # analyze_custom_input_review_aspect(custom_review, X, y)


if __name__ == '__main__':
    main()

