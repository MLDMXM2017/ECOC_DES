import pandas as pd
import numpy as np
import os
import random
import math
from tkinter import _flatten

"""
Split data

Modify:
    2021-1-16
"""
def train_valid_test_split(names, in_dir, out_dir, exp_num):
    print("split data...")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    for idx in range(exp_num):
        for dn in names:
            i_f = '%s%s.csv' % (in_dir, dn)
            X0, y0 = _read_data(i_f)
            unique_y = np.unique(y0)
            labels = [y0[np.where(y0==label)] for label in unique_y]
            y_nums = [len(labels[i]) for i in range(len(labels))]
            select_y = []
            for i in range(len(y_nums)):
                if y_nums[i] > 20:
                    select_y.append(unique_y[i])
            X = []
            y = []
            for i in range(len(y0)):
                for sel_y in select_y:
                    if y0[i] == sel_y:
                        X.append(X0[i])
                        y.append(y0[i])
                        break
            X = np.array(X)
            y = np.array(y)
            y_label = np.array([y]).T
            new_data = np.hstack((X,y_label))
            new_o_f = 'data/source/%s.csv' % (dn)
            np.savetxt(new_o_f, new_data, fmt='%s', delimiter=',')
            
            classes=np.unique(y)
            all_class_num=len(classes)
            m=len(X)
            dict={}
            X_ = []
            y_ = []
            for i in range(all_class_num):
                X_.append([])
                y_.append([])
            for i in range(m):
                X_[np.where(classes==y[i])[0][0]].append(X[i])
                y_[np.where(classes==y[i])[0][0]].append(y[i])
                if y[i] in dict:
                    dict[y[i]]+=1
                else:
                    dict[y[i]]=1
            class_num=[]
            for i in range(all_class_num):
                class_num.append(dict[classes[i]])
            set_num=[]
            train_class_num=[];valid_class_num=[];test_class_num=[]
            for i in range(all_class_num):
                train_class_num.append(math.ceil(class_num[i]*0.6))
                valid_class_num.append(int(class_num[i]*0.2))
                test_class_num.append(int(class_num[i]-math.ceil(class_num[i]*0.6)-int(class_num[i]*0.2)))
            set_num.append(train_class_num)
            set_num.append(valid_class_num)
            set_num.append(test_class_num)
            trainMat=[];trainClasses=[];validMat=[];validClasses=[];testMat=[];testClasses=[]
            for i in range(all_class_num):
                testIndex=list(range(0,len(X_[i])));validIndex=[];trainIndex=[]
                for j in range(set_num[0][i]):
                    randIndex=int(random.uniform(0,len(testIndex)))
                    trainIndex.append(testIndex[randIndex])
                    del(testIndex[randIndex])
                for j in range(set_num[1][i]):
                    randIndex=int(random.uniform(0,len(testIndex)))
                    validIndex.append(testIndex[randIndex])
                    del(testIndex[randIndex])
                for j in trainIndex:
                    trainMat.append(X_[i][j])
                    trainClasses.append(y_[i][j])
                for j in validIndex:
                    validMat.append(X_[i][j])
                    validClasses.append(y_[i][j])
                for j in testIndex:
                    testMat.append(X_[i][j])
                    testClasses.append(y_[i][j])
            train_X=np.array(trainMat)
            valid_X=np.array(validMat)
            test_X=np.array(testMat)
            train_y=np.array(trainClasses)
            valid_y=np.array(validClasses)
            test_y=np.array(testClasses)
            train_data=np.column_stack((train_X,train_y))
            valid_data=np.column_stack((valid_X,valid_y))
            test_data=np.column_stack((test_X,test_y))
            train_o_f='%s%s_%d_train.csv' % (out_dir, dn, idx)
            valid_o_f='%s%s_%d_valid.csv' % (out_dir, dn, idx)
            test_o_f='%s%s_%d_test.csv' % (out_dir, dn, idx)
            np.savetxt(train_o_f, train_data, fmt='%s', delimiter=',')
            np.savetxt(valid_o_f, valid_data, fmt='%s', delimiter=',')
            np.savetxt(test_o_f, test_data, fmt='%s', delimiter=',')

def _read_data(f):
    X = np.loadtxt(f, np.str, delimiter=',')
    y = X[:, -1]
    X = X[:, :-1].astype(np.float)
    return X, y
