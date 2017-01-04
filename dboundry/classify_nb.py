from sklearn.naive_bayes import GaussianNB

def classify(features_train, labels_train):
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    return clf

def accuracy(features_train, labels_train, features_test, labels_test):
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    return clf.score(features_test, labels_test)
