from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def get_cly(bcn):
    if bcn == 'SVM':    
        cly = SVC()
    elif bcn == 'KNN':
        cly = KNeighborsClassifier()
    elif bcn == 'DTree':
        cly = tree.DecisionTreeClassifier()
    elif bcn == 'Bayes':
        cly = GaussianNB()
    elif bcn == 'Logi':
        cly = LogisticRegression()
    elif bcn == 'NN':
        cly = MLPClassifier()
    else:
        print('Base classifier name error')
    return cly

def cly_data(X_trn, y_trn, X_vld, y_vld, bcn):
    if bcn == 'SVM':    
        cly = SVC()
    elif bcn == 'KNN':
        cly = KNeighborsClassifier()
    elif bcn == 'DTree':
        cly = tree.DecisionTreeClassifier()
    elif bcn == 'Bayes':
        cly = GaussianNB()
    elif bcn == 'Logi':
        cly = LogisticRegression()
    elif bcn == 'NN':
        cly = MLPClassifier()
    else:
        print('Base classifier name error')
    cly.fit(X_trn, y_trn)
    return accuracy_score(cly.predict(X_vld), y_vld)
    
def cly_data_(X_trn, y_trn, X_vld, y_vld, bcn):
    if bcn == 'SVM':    
        cly = SVC()
    elif bcn == 'KNN':
        cly = KNeighborsClassifier()
    elif bcn == 'DTree':
        cly = tree.DecisionTreeClassifier()
    elif bcn == 'Bayes':
        cly = GaussianNB()
    elif bcn == 'Logi':
        cly = LogisticRegression()
    elif bcn == 'NN':
        cly = MLPClassifier()
    else:
        print('Base classifier name error')
    cly.fit(X_trn, y_trn)
    return accuracy_score(cly.predict(X_vld), y_vld), cly