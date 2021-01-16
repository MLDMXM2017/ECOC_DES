import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import math
from Tools.Gaussia import GDA

def fisher_measure(X_train, y_train, X_test, y_code):
    """Fisher measure"""
    if check_X(X_train[y_train!=0], y_train[y_train!=0]):
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train[y_train!=0], np.array([y_train[y_train!=0]]).T)
        test_proba = np.array(lda.predict_proba(X_test))

        m = lda.means_            
        w = lda.coef_             
        m_ = np.dot(m, w.T)       
        m0 = m.mean(axis=0)       
        y0 = np.dot(m0, w.T)      
        Y = np.dot(X_test, w.T)   
        dis = abs(Y - y0)         
        for i in range(len(dis)):       
            dis[i] = 1.0/(1.0 + np.exp(-dis[i])) - 0.5    
            dis[i] = dis[i] * 2                           
        return dis.T[0]
    else:
        score = []
        for i in range(len(X_test)):
            score.append(1.)
        return score
    
def fisher_gaussia_measure(X_train, y_train, X_test, y_code):
    """Fisher measure combined with gaussia probability density model"""
    if check_X(X_train[y_train!=0], y_train[y_train!=0]):
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train[y_train!=0], np.array([y_train[y_train!=0]]).T)

        m = lda.means_            
        w = lda.coef_             
        m_ = np.dot(m, w.T)       
        m0 = m.mean(axis=0)       
        y0 = np.dot(m0, w.T)      
        
        X = np.dot(X_train[y_train!=0], w.T)    
        x_p = np.dot(X_train[y_train==1], w.T)  
        x_n = np.dot(X_train[y_train==-1], w.T) 
        y = y_train[y_train!=0]
        
        gda = GDA(X, y)
        point = gda.find_intersection(X)
        
        X_t = np.dot(X_test, w.T) 
        dis = abs(X_t - point)               
        for i in range(len(dis)):       
            dis[i] = 1.0/(1.0 + np.exp(-dis[i])) - 0.5    
            dis[i] = dis[i] * 2                           
        return dis.T[0]
    else:
        score = []
        for i in range(len(X_test)):
            score.append(1.)
        return score
    
def check_X(X, y):
    sample_num = len(X)
    flag_one = False
    flag_two = False
    flag_three = False
    i = X[0][0]
    for X_ in X:
        for x in X_:
            if x != i:
                flag_one = True
    j = X[y==1][0][0]
    for X_ in X[y==1]:
        for x in X_:
            if x != i:
                flag_two = True
    j = X[y==-1][0][0]
    for X_ in X[y==-1]:
        for x in X_:
            if x != i:
                flag_three = True
    if flag_one == True:
        if flag_two == True:
            if flag_three == True:
                return True
    return False