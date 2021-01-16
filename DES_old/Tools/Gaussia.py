import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib as mpl
import matplotlib.pyplot as plt

class GDA:
    def __init__(self,train_data,train_label):
        """
        This is the GDA algorithm constructor
        :param train_data:  training data
        :param train_label: training label
        """
        self.Train_Data = train_data
        self.Train_Label = train_label
        self.postive_num = 0   
        self.negetive_num = 0  
        postive_data = []         
        negetive_data = []      
        for (data,label) in zip(self.Train_Data,self.Train_Label):
            if label == 1:          # positive sample
                self.postive_num += 1
                postive_data.append(list(data))
            else:                   # negative sample
                self.negetive_num += 1
                negetive_data.append(list(data))
        # Calculate the probability of binomial distribution of positive and negative samples
        row,col = np.shape(train_data)
        self.postive = self.postive_num*1.0/row
        self.negetive = 1-self.postive
        # Calculate the mean vector of Gaussian distribution of positive and negative samples
        postive_data = np.array(postive_data)
        negetive_data = np.array(negetive_data)
        postive_data_sum = np.sum(postive_data, 0)
        negetive_data_sum = np.sum(negetive_data, 0)
        self.mu_positive = postive_data_sum*1.0/self.postive_num 
        self.mu_negetive = negetive_data_sum*1.0/self.negetive_num
        # Calculating the covariance matrix of Gaussian distribution
        positive_deta = postive_data-self.mu_positive
        negetive_deta = negetive_data-self.mu_negetive
        self.sigma = []
        for deta in positive_deta:
            deta = deta.reshape(1,col)
            ans = deta.T.dot(deta)
            self.sigma.append(ans)
        for deta in negetive_deta:
            deta = deta.reshape(1,col)
            ans = deta.T.dot(deta)
            self.sigma.append(ans)
        self.sigma = np.array(self.sigma)
        self.sigma = np.sum(self.sigma,0)
        self.sigma = self.sigma/row
        self.mu_positive = self.mu_positive.reshape(1,col)
        self.mu_negetive = self.mu_negetive.reshape(1,col)

    def Gaussian(self, x, mean, cov):
        """
        This is the Gaussian probability density function
        :param x:    input data
        :param mean: mean vector
        :param cov:  covariance matrix
        :return:     probability of x
        """
        dim = np.shape(cov)[0]
        # Calculate the determinant and inverse matrix of cov
        covdet = np.linalg.det(cov + np.eye(dim) * 0.001)
        covinv = np.linalg.inv(cov + np.eye(dim) * 0.001)
        xdiff = (x - mean).reshape((1, dim))
        # probability density
        prob = 1.0 / (np.power(np.power(2 * np.pi, dim) * np.abs(covdet), 0.5)) * \
               np.exp(-0.5 * xdiff.dot(covinv).dot(xdiff.T))[0][0]
        return prob

    def predict(self,test_data):
        """Predict input data"""
        predict_label = []
        for data in test_data:
            positive_pro = self.Gaussian(data,self.mu_positive,self.sigma)
            negetive_pro = self.Gaussian(data,self.mu_negetive,self.sigma)
            if positive_pro >= negetive_pro:
                predict_label.append(1)
            else:
                predict_label.append(0)
        return predict_label
        
    def predict_proba(self,test_data):
        """Calculate the probability of input data"""
        predict_label = []
        for data in test_data:
            positive_pro = self.Gaussian(data,self.mu_positive,self.sigma)
            negetive_pro = self.Gaussian(data,self.mu_negetive,self.sigma)
            if positive_pro >= negetive_pro:
                predict_label.append(positive_pro)
            else:
                predict_label.append(negetive_pro)
        return predict_label
        
    def find_intersection(self, train_data):
        """Find the intersection of positive and negative cluster"""
        data = sorted(train_data)
        left = data[0]
        right = data[-1]
        step = (right-left)/100.
        dis = []
        if right - left == 0:
            return np.array([0.])
        for i in np.arange(left, right, step):
            positive_pro = self.Gaussian(i,self.mu_positive,self.sigma)
            negetive_pro = self.Gaussian(i,self.mu_negetive,self.sigma)
            dis.append(positive_pro - negetive_pro)
        point = (left+right)/2.0
        h = []
        for i in range(len(dis)-1):
            if dis[i]*dis[i+1] < 0:
                point = (dis[i]+dis[i+1])/2.0
                h.append(point)
        return np.array([point])
