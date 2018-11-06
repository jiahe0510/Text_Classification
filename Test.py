import Readfile
import Models
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

path1 = "neg.tok"
path2 = "pos.tok"

r = Readfile.Readfile(path1, path2, index=2)
model = Models.Models()


data_set = model.binary_feature(r.data_set)
np.random.shuffle(data_set)

print(data_set.shape)
row, col = data_set.shape
X = data_set[:, :col-1]
y = data_set[:, col-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print(X_train.shape)
print(y_train.shape)

clf = MultinomialNB()
cv_results = cross_val_score(clf, X_train, y_train.ravel(), scoring='accuracy', cv=10)
print('Accuracy: ', cv_results)

clf.fit(X_train, y_train)
target_names = ['negative', 'positive']
y_pred = clf.predict(X_test)
print(classification_report(y_pred, y_test, target_names=target_names))
print(confusion_matrix(y_pred, y_test, labels=[1, 0]))

lgc = LogisticRegression(solver='lbfgs')
lgc_results = cross_val_score(lgc, X_train, y_train.ravel(), scoring='accuracy', cv=10)
print('Accuracy for logistic regression:', lgc_results)
lgc.fit(X_train, y_train)
target_names = ['negative', 'positive']
y_pred = lgc.predict(X_test)
print(classification_report(y_pred, y_test, target_names=target_names))
print(confusion_matrix(y_pred, y_test, labels=[1, 0]))