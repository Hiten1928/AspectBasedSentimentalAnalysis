import pandas as pd
import codecs
from sklearn import svm
import csv
import nltk
from nltk.tag.stanford import StanfordPOSTagger as POS_Tag
from nltk import word_tokenize
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import warnings

def read_train_reviews_file(path):
    text_list = []
    opinion_list = []
    with open(path, "r") as file:
        next(file)
        reader = csv.reader(file)

        for row in reader:
            opinion_inner_list = []
            text_list.append(row[4])
            opinion_dict = {
                row[2]: row[1]
            }
            opinion_inner_list.append(opinion_dict)
            opinion_list.append(opinion_inner_list)
    return text_list, opinion_list


def read_train_file(path):
    text_list = []
    opinion_list = []
    types_of_encoding = ["utf8", "cp1252"]
    for encoding_type in types_of_encoding:
        with codecs.open(path, encoding=encoding_type, errors='replace') as file:
            next(file)
            reader = csv.reader(file)

            for row in reader:
                opinion_inner_list = []
                text_list.append(row[4])
                opinion_dict = {
                    row[2]: row[1]
                }
                opinion_inner_list.append(opinion_dict)
                opinion_list.append(opinion_inner_list)
            return text_list, opinion_list

def get_most_common_aspect(opinion_list):
    opinion= []
    for inner_list in opinion_list:
        for _dict in inner_list:
            for key in _dict:
                opinion.append(key)
    most_common_aspect = [k for k,v in nltk.FreqDist(opinion).most_common(2)]
    return most_common_aspect

def posTag(review):
    _path_to_model = 'C:\\Users\ROSS\Documents\Study\Software Quality\stanford-postagger-2018-02-27\models\english-bidirectional-distsim.tagger'
    _path_to_jar = 'C:\\Users\ROSS\Documents\Study\Software Quality\stanford-postagger-2018-02-27\stanford-postagger-3.9.1.jar'
    stanford_tag = POS_Tag(model_filename=_path_to_model, path_to_jar=_path_to_jar)
    tagged_text_list = []
    for text in review:
        tagged_text_list.append(stanford_tag.tag(word_tokenize(text)))
    return tagged_text_list


def filterTag(tagged_review):
    final_text_list=[]
    for text_list in tagged_review:
        final_text=[]
        for word,tag in text_list:
            if tag in ['NN','NNS','NNP','NNPS','RB','RBR','RBS','JJ','JJR','JJS','VB','VBD','VBG','VBN','VBP','VBZ']:
                final_text.append(word)
        final_text_list.append(' '.join(final_text))
    return final_text_list


def get_data_frame(text_list, opinion_list, most_common_aspect):
    data = {'Review': text_list}
    df = pd.DataFrame(data)
    if opinion_list:
        for inner_list in opinion_list:
            for _dict in inner_list:
                for key in _dict:
                    if key in most_common_aspect:
                        df.loc[opinion_list.index(inner_list), key] = _dict[key]
    return df


def get_aspect_data_frame(df,most_common_aspect):
    for common_aspect in most_common_aspect:
        df[common_aspect] = df[common_aspect].replace(['positive', 'negative', 'neutral'], [1, 1, 1])
    df = df.fillna(0)
    return df


def create_train_df(train_path):
    train_text_list, train_opinion_list = read_train_reviews_file(train_path)
    # print(text_list)
    # print(opinion_list)
    most_common_aspect = get_most_common_aspect(train_opinion_list)
    # print(most_common_aspect)
    tagged_text_list_train = posTag(train_text_list)
    # print(tagged_text_list_train)
    joblib.dump(tagged_text_list_train, 'tagged_text_list_train.pkl')
    train_tagged_text_list = joblib.load('tagged_text_list_train.pkl')
    # print(train_tagged_text_list)
    final_train_text_list = filterTag(tagged_text_list_train)
    # print(final_train_text_list)
    df_train = get_data_frame(final_train_text_list, train_opinion_list, most_common_aspect)
    # print("df_train")
    # print(df_train)
    df_train_aspect = get_aspect_data_frame(df_train, most_common_aspect)
    # print("df_train_aspect")
    # print(df_train_aspect)
    df_train_aspect = df_train_aspect.reindex_axis(sorted(df_train_aspect.columns), axis=1)
    # print("df_train_aspect")
    # print(df_train_aspect)
    return df_train_aspect


def create_test_df(test_path):
    test_text_list, test_opinion_list = read_train_file(test_path)
    most_common_aspect = get_most_common_aspect(test_opinion_list)
    tagged_text_list_test=posTag(test_text_list)
    joblib.dump(tagged_text_list_test, 'tagged_text_list_test.pkl')
    tagged_text_list_test = joblib.load('tagged_text_list_test.pkl')
    final_test_text_list = filterTag(tagged_text_list_test)
    df_test = get_data_frame(final_test_text_list, test_opinion_list, most_common_aspect)
    df_test_aspect = get_aspect_data_frame(df_test, most_common_aspect)
    df_test_aspect = df_test_aspect.reindex_axis(sorted(df_test_aspect.columns), axis=1)
    return df_test_aspect


def main():
    train_path = "C:\\Users\ROSS\Documents\Study\Software Quality\Project\Game_Review_Train.csv"
    df_train_aspect = create_train_df(train_path)
    #print(df_train_aspect)
    test_path = "C:\\Users\ROSS\Documents\Study\Software Quality\Project\Game_Review_Train.csv"
    df_test_aspect = create_test_df(test_path)
    #print(df_test_aspect)

    # Sort the data frame according to aspect's name and separate data(X) and target(y)
    #df_train_aspect = df_train_aspect.sample(frac=1).reset_index(drop=True) #For randoming
    X_train = df_train_aspect.Review
    y_train = df_train_aspect.drop('Review', 1)
    # print(X_train)
    # print(y_train)

    #df_test_aspect = df_test_aspect.sample(frac=1).reset_index(drop=True) #For randoming
    X_test = df_test_aspect.Review
    y_test = df_test_aspect.drop('Review', 1)
    # print(X_test)
    # print(y_test)

    # Change y_train to numpy array
    import numpy as np
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    # Generate word vecotors using CountVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    from nltk import word_tokenize
    from nltk.stem import WordNetLemmatizer
    vect = CountVectorizer(max_df=1.0, stop_words='english')
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)

    # Create various models. These are multi-label models.
    nb_classif = OneVsRestClassifier(MultinomialNB()).fit(X_train_dtm, y_train)
    C = 1.0  # SVregularization parameter
    svc = OneVsRestClassifier(svm.SVC(kernel='linear', C=C)).fit(X_train_dtm, y_train)
    lin_svc = OneVsRestClassifier(svm.LinearSVC(C=C)).fit(X_train_dtm, y_train)
    sgd = OneVsRestClassifier(SGDClassifier()).fit(X_train_dtm, y_train)

    # Predict the test data using classifiers
    y_pred_class = nb_classif.predict(X_test_dtm)
    y_pred_class_svc = svc.predict(X_test_dtm)
    y_pred_class_lin_svc = lin_svc.predict(X_test_dtm)
    y_pred_class_sgd = sgd.predict(X_test_dtm)

    # Following code to test metrics of all aspect extraction classifiers
    print("NB Classifier Accuracy-Precision-Recall-Fscore")
    print(metrics.accuracy_score(y_test, y_pred_class))
    print(metrics.precision_score(y_test, y_pred_class, average='micro'))
    print(metrics.recall_score(y_test, y_pred_class, average='micro'))
    print(metrics.f1_score(y_test, y_pred_class, average='micro'))

    print("DTM Accuracy-Precision-Recall-Fscore")
    print(metrics.accuracy_score(y_test, y_pred_class_svc))
    print(metrics.precision_score(y_test, y_pred_class_svc, average='micro'))
    print(metrics.recall_score(y_test, y_pred_class_svc, average='micro'))
    print(metrics.f1_score(y_test, y_pred_class_svc, average='micro'))

    print("linear SVC Accuracy-Precision-Recall-Fscore")
    print(metrics.accuracy_score(y_test, y_pred_class_lin_svc))
    print(metrics.precision_score(y_test, y_pred_class_lin_svc, average='micro'))
    print(metrics.recall_score(y_test, y_pred_class_lin_svc, average='micro'))
    print(metrics.f1_score(y_test, y_pred_class_lin_svc, average='micro'))

    print("class_sgd Accuracy-Precision-Recall-Fscore")
    print(metrics.accuracy_score(y_test, y_pred_class_sgd))
    print(metrics.precision_score(y_test, y_pred_class_sgd, average='micro'))
    print(metrics.recall_score(y_test, y_pred_class_sgd, average='micro'))
    print(metrics.f1_score(y_test, y_pred_class_sgd, average='micro'))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(metrics.classification_report(y_test, y_pred_class))
        print(metrics.classification_report(y_test, y_pred_class_svc))
        print(metrics.classification_report(y_test, y_pred_class_lin_svc))
        print(metrics.classification_report(y_test, y_pred_class_sgd))


if __name__ == '__main__':
   main()
