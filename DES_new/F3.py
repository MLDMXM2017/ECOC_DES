import os
import time
import math
import numpy as np
from Classifiers.BaseClassifier import get_base_clf
from Classifiers.ECOCClassifier import SimpleECOCClassifier, predict_binary
from DataComplexity.Get_Complexity import get_complexity_F1, get_complexity_F2, get_complexity_F3, get_complexity_N2, get_complexity_N3, get_complexity_N4, get_complexity_L3, get_complexity_Cluster, get_complexity_D2
from Decoding.Decoder import get_decoder
from Tools.Distance import fisher_measure, fisher_gaussia_measure
from Tools.FeatureSelect import fea_slt_number, feature_select
from Tools.Matrix import create_matrix
from Tools.Norm import normalize_data
from Tools.Read import read_data
from Tools.Split import train_valid_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

def do_exp(idx, bcn, fname, feature_space, matrix_code):
    """2021-1-16 Experiments on UCI data sets"""
    if not os.path.exists(out_dir):
        print('Creating %s' % out_dir)
        os.mkdir(out_dir)
    
    print('%s %s %s %d starts...' % (fname, bcn, matrix_code, idx))
    data_set = {}
    point_num_test = -1 
    mat_len = -1    
    y_names, y_index = None, None   
    out_file = '%s%s_%s_%s_%d_res.csv' % (out_dir, bcn, fname, matrix_code, idx)
    if os.path.exists(out_file):
        print('%s exist!' % out_file)
        return None  

    mat_file = 'data/exp_mat/%s_%s_%d.csv' % (fname, matrix_code, idx)
    mat = np.loadtxt(mat_file, np.int, delimiter=',')
    for fs in feature_space:
        if mat_len == -1:
            mat_len = mat.shape[1]
        elif mat.shape[1] != mat_len:
            raise ValueError('The length of matrix is not the same.')
        
        for col_i in range(mat_len):
            s_col_i = str(col_i) 
            train_file = '%s%s_%s_%s_%s_%d_%d_train.csv' % (dat_dir, bcn, fname, matrix_code, fs, idx, col_i)
            X_train, y_train = read_data(train_file)
            valid_file = '%s%s_%s_%s_%s_%d_%d_valid.csv' % (dat_dir, bcn, fname, matrix_code, fs, idx, col_i)
            X_valid, y_valid = read_data(valid_file)
            test_file = '%s%s_%s_%s_%s_%d_%d_test.csv' % (dat_dir, bcn, fname, matrix_code, fs, idx, col_i)
            X_test, y_test = read_data(test_file)
            if y_names is None:
                y_names = np.unique(y_train)
                y_index = dict((c,i) for i,c in enumerate(y_names))
            len_train, len_test = len(y_train), len(y_test)
            if point_num_test == -1:
                point_num_test = len_test
            elif point_num_test != len_test:
                raise ValueError('Different length of test data.')
            
            y_code = mat[:, col_i]
            y_trn_ = np.array([y_code[y_index[y]] for y in y_train])
            y_vld_ = np.array([y_code[y_index[y]] for y in y_valid])
            y_tst_ = np.array([y_code[y_index[y]] for y in y_test])
            
            #Training ECOC classifier
            ecoc = SimpleECOCClassifier(get_base_clf(bcn), mat, decoder_code)
            ecoc.fit(X_train, y_train)
            pred_valid = ecoc.predict_(X_valid[y_vld_!=0], col_i)
            estimators = ecoc.estimators_[col_i]
            
            #F3
            positive_X = X_train[y_trn_==1]
            positive_y = y_trn_[y_trn_==1]
            negative_X = X_train[y_trn_==-1]
            negative_y = y_trn_[y_trn_==-1]
            weight = get_complexity_F3(positive_X, positive_y, negative_X, negative_y)
            print('%s %d: %f' % (fs, col_i, weight))

            data_set[fs+s_col_i] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_valid': X_valid,
                'y_valid': y_valid,
                'X_test': X_test,
                'y_test': y_test,
                'len_train': len_train,
                'len_test': len_test,
                'estimator': estimators,
                'weight': weight
            }       
    
    distances = []
    for col_i in range(mat_len):
        s_col_i = str(col_i)
        dis_col = []
        for fs in feature_space:
            d_i = feature_space.index(fs)
            y_code = mat[:, col_i]
            y_trn_ = np.array([y_code[y_index[y]] for y in data_set[fs+s_col_i]['y_train']])
            y_tst_ = np.array([y_code[y_index[y]] for y in data_set[fs+s_col_i]['y_test']])
            dis_col.append(fisher_gaussia_measure(data_set[fs+s_col_i]['X_train'], y_trn_, data_set[fs+s_col_i]['X_test'], y_code))
        distances.append(dis_col)
    distances = np.array(distances)

    pred = []
    for test_i in range(point_num_test):
        sample_analyses = 'Sample %d/%d, label %s' % (test_i+1, point_num_test, y_test[test_i])
        print(sample_analyses)
        
        matrix_, fstags_, estimators_, classifier_weight = [], [], [], []
        distance = distances[:, :, test_i]
        for col_i in range(mat_len):
            s_col_i = str(col_i)
            dis_ = distance[col_i, :]
            minus_weights = np.array([data_set[k+s_col_i]['weight'] for k in feature_space])    # 得到每个fs的权重
            score_ = np.array(dis_) * minus_weights
            fs_ = feature_space[score_.argmax()]
            classifier_weight.append(data_set[fs_+s_col_i]['weight'])
            fstags_.append(fs_)
            matrix_.append(mat[:, col_i])
            estimators_.append(data_set[fs_+s_col_i]['estimator'])
        classifier_weight = np.array([classifier_weight])
        matrix_ = np.array(matrix_).T
        Y = np.array([predict_binary(estimators_[i], data_set[fstags_[i]+str(i)]['X_test'][[test_i]]) for i in
                        range(len(estimators_))]).T
        dd_ = decoder.decode(Y * classifier_weight[0], matrix_)
        p_ = dd_.argmin(axis=1)[0]
        pred.append(y_names[p_])
    """
    The output file structure is as follows：
                pred    true
    sample1     C1      C1
    sample2     C2      C1
    sample3     C2      C3
    ...         ...     ...
    accuracy    90%    100%
    """
    test_file = '%s%s_%s_%s_%s_%d_%d_test.csv' % (dat_dir, bcn, fname, matrix_code, fs, idx, 0)
    X_test, y_test = read_data(test_file)
    pred_col = ['ensemble', 'real']
    acc = [round(accuracy_score(pred, y_test), 4), 1.]  #保留四位小数
    pred = np.array([pred, y_test]).T
    pred = np.r_['0,2', pred_col, pred, acc]
    np.savetxt(out_file, pred, delimiter=',', fmt='%s')
    
if __name__ == '__main__':
    exp_num = 10                             #Number of experiments
    # File path
    src_dir = 'data/source/'                #Storing source data
    spl_dir = 'data/split/'                 #Storing splited data
    nor_dir = 'data/norm/'                  #Storing normalized data
    mat_dir = 'data/exp_mat/'               #Storing coding matrix
    spt_dir = 'data/support/'               #Storing feature score
    fea_dir = 'data/fea_num/'               #Storing feature selected number
    dat_dir = 'data/exp_data/'              #Storing experimental data
    out_dir = 'data/exp/'                   #Storing experimental result
    
    # Get decoder
    decoder_code = 'AED'
    decoder = get_decoder(decoder_code)
    
    suffix = ['_train.csv', '_valid.csv', '_test.csv']
    feature_space = ['bw', 'chi2', 'kf', 'kmi', 'var']  #Feature selection methods
    file_names = ['car', 'dermatology', 'flare', 'led24digit', 'mfeat', 'nursery', 'optdigits', 'pendigits', 'satimage', 'segment', 'shuttle', 'splice', 'texture', 'vehicle', 'vowel']  #我先不加isolet和letter了                                                                #总数据集       flore与zoo只剩下了一类
                                                        #UCI data
    matrix_types = ['OVA', 'OVO', 'DEcoc', 'DenseRand', 'SparseRand', 'DC_ECOC', 'ECOC_ONE']
                                                        #ECOC algorithms
    base_cly_names = ['SVM', 'Logi', 'KNN', 'DTree', 'Bayes']
                                                        #Base classifiers

    # Segmentation of data into train:validation:test=3:1:1
    train_valid_test_split(file_names, src_dir, spl_dir, exp_num)
    # MinMax normalize data
    normalize_data(file_names, spl_dir, nor_dir, exp_num)
    # Generating coding matrix for data
    create_matrix(file_names, src_dir, mat_dir, matrix_types, exp_num)
    
    for base in base_cly_names:
        # Determine the size of feature subsets
        fea_slt_number(base, file_names, feature_space, matrix_types, nor_dir, mat_dir, spt_dir, fea_dir, exp_num)
        # Select feature
        feature_select(base, file_names, feature_space, matrix_types, nor_dir, mat_dir, spt_dir, fea_dir, dat_dir, exp_num)
        for idx in range(exp_num):
            for fn in file_names:
                for mc in matrix_types:
                    start = time.time()
                    do_exp(idx, base, fn, feature_space, mc)
                    end = time.time()
                    print('%s_%s_%s_%d: %f' % (base, fn, mc, idx, end-start))
        