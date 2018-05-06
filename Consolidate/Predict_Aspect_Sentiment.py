from Sentiment.Predict_Sentiment import analyze_custom_input_review_sentiment, create_sentiments_pandas_frame, create_sentiments_test_train_set
from Aspect.Predict_Aspect import analyze_custom_input_review_aspect, create_aspects_pandas_frame, create_aspect_test_train_set


#process data to create dataframes with polarity and category/aspect
processed_sentiments_data = create_sentiments_pandas_frame()
processed_aspects_data = create_aspects_pandas_frame()

#create train and test set as input for training the classifier
X_sent, y_sent, X_train_sent, X_test_sent, y_train_sent, y_test_sent, X_train_dtm_sent, X_test_dtm_sent, vectorizer_sent = \
        create_sentiments_test_train_set(processed_sentiments_data)
X_aspect, y_aspect, X_train_aspect, X_test_aspect, y_train_aspect, y_test_aspect, X_train_dtm_aspect, X_test_dtm_aspect, vectorizer_aspect = \
        create_aspect_test_train_set(processed_aspects_data)

custom_review = []
print("**************************************************************************************************************")
print('Enter review to predict its aspect & sentiment: ', end=" ")
custom_review.append(input())
analyze_custom_input_review_sentiment(custom_review, X_sent, y_sent) #predict the review sentiment
print("and")
analyze_custom_input_review_aspect(custom_review, X_aspect, y_aspect) #predict the review aspect

