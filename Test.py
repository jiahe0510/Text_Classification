import Readfile
import Models
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

path1 = "neg.tok"
path2 = "pos.tok"

r_unigram = Readfile.Readfile(path1, path2, index=1)
r_bigram = Readfile.Readfile(path1, path2, index=2)
data_unigram = r_unigram.data_set
data_bigram = r_bigram.data_set
model = Models.Models()


def test_model(data, classifier, message='-'):
    print('-------------------%s below--------------------' % message)
    data_set = model.binary_feature(data)
    np.random.shuffle(data_set)
    row, col = data_set.shape
    X = data_set[:, :col - 1]
    y = data_set[:, col - 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    cv_results = cross_validate(classifier, X_train, y_train.ravel(), return_train_score=False,
                                scoring=('accuracy', 'f1', ), cv=10)
    print('Accuracy: ', cv_results['test_accuracy'])
    print('f1: ', cv_results['test_f1'])
    print('Mean accuracy of cross validation is:', cv_results['test_accuracy'].mean())
    print('Mean f1 of cross validation is: ', cv_results['test_f1'].mean())

    classifier.fit(X_train, y_train)
    target_names = ['negative', 'positive']
    y_pred = classifier.predict(X_test)
    print(classification_report(y_pred, y_test, target_names=target_names))
    print(confusion_matrix(y_pred, y_test, labels=[1, 0]))
    print('-------------------%s above--------------------' % message)


# Test Naive Bayes using Uni-gram, Bi-gram bag-of-words:
clf = MultinomialNB()
test_model(data_unigram, clf, message='Uni-gram Naive Bayes')
test_model(data_bigram, clf, message='Bi-gram Naive Bayes')

# Test Logistic Regression using Unigram, Bigram bag-of-words:
lgr = LogisticRegression(solver='lbfgs')
test_model(data_unigram, lgr, message='Uni-gram Logistic Regression')
test_model(data_bigram, lgr, message='Bi-gram Logistic Regression')

ratio_unigram = model.combine_bayes_logistic(data_unigram)
ration_bigram = model.combine_bayes_logistic(data_bigram)

data_unigram_ratio = model.process_vector_with_ratio(data_unigram, ratio_unigram)
data_bigram_ratio = model.process_vector_with_ratio(data_bigram, ration_bigram)

test_model(data_unigram_ratio, lgr, message='Uni-gram Logistic Regression with Naive Bayes Feature')
test_model(data_bigram_ratio, lgr, message='Bi-gram Logistic Regression with Naive Bayes Feature')


