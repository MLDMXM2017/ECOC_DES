import os
import time
import math
import numpy as np
from Tools.Read import read_data
from Tools.Dichotomy import cly_data
from FeatureSelection.GetFS import fea_rank
import matplotlib.pyplot as plt

def fea_slt(i_f, t_f, o_f, k):
    """Select feature"""
    spt = np.loadtxt(t_f, np.int, delimiter=',')
    X, y = read_data(i_f)
    X_new = X[:, np.sort(spt[:k])]
    X_new = np.c_[X_new, y]
    np.savetxt(o_f, X_new, fmt='%s', delimiter=',')

def find_peak(accs, fea_step, fea_num, beta):
    """Find the optimal number of feature"""
    num = len(accs)
    cmp_num = math.ceil(num * beta)+1
    cmp_acc = []
    for i in range(0, cmp_num):
        cmp_acc.append(accs[i])
    max_i = cmp_acc.index(max(cmp_acc))
    for i in range(cmp_num, num):
        if accs[i] >= min(cmp_acc):
            if accs[i] > max(cmp_acc):            
                max_i = i
            del cmp_acc[cmp_acc.index(min(cmp_acc))]
            cmp_acc.append(accs[i])
            continue
        else:
            break
    if (max_i + 1) * fea_step > fea_num:
        return fea_num, max(cmp_acc)
    else:
        return (max_i + 1) * fea_step, max(cmp_acc)
                    
def fea_slt_number(bcn, file_names, fs_tags, matrix_types, nor_dir, mat_dir, spt_dir, fea_dir, exp_num):
    """Determine the size of feature subsets"""
    if not os.path.exists(spt_dir):
        os.mkdir(spt_dir)
    if not os.path.exists(fea_dir):
        os.mkdir(fea_dir)
    
    alpha=0.05  # Iteration Step
    beta=0.02   # Size of comparison interval
    for idx in range(exp_num):
        for fn in file_names:
            X_trn, y_trn = read_data('%s%s_%d_train.csv' % (nor_dir, fn, idx))
            X_vld, y_vld = read_data('%s%s_%d_valid.csv' % (nor_dir, fn, idx))
            fea_num = X_trn.shape[1]
            fea_step = math.ceil(fea_num * alpha)
            fea_k_list = list(range(fea_step, fea_num, fea_step))
            
            y_names = np.unique(y_trn)
            y_index = dict((c,i) for i,c in enumerate(y_names))
            for mc in matrix_types:
                print('processing %s %s %d...' % (fn, mc, idx))
                mat_file = '%s%s_%s_%d.csv' % (mat_dir, fn, mc, idx)
                mat = np.loadtxt(mat_file, np.int, delimiter=',')   
                mat_len = mat.shape[1]
                for col_i in range(mat_len):
                    fea_file = '%s%s_%s_%s_%d_%d_fea_num.csv' % (fea_dir, bcn, fn, mc, idx, col_i)
                    y_code = mat[:, col_i]
                    ytrn_ = np.array([y_code[y_index[y]] for y in y_trn])
                    yvld_ = np.array([y_code[y_index[y]] for y in y_vld])
                    X_train = X_trn[ytrn_!=0]
                    X_valid = X_vld[yvld_!=0]
                    y_train = ytrn_[ytrn_!=0]
                    y_valid = yvld_[yvld_!=0]
                    
                    clock_start = time.time()
                    fs_nums = []
                    fs_accs = []
                    for fs in fs_tags:
                        spt_file = '%s%s_%s_%s_%s_%d_%d.csv' % (spt_dir, bcn, fn, mc, fs, idx, col_i)
                        rank_ind = fea_rank(X_train, y_train, fs)
                        np.savetxt(spt_file, rank_ind.T, delimiter=',', fmt='%d')
                        fea = np.loadtxt(spt_file, delimiter=',', dtype=np.int)
                        accs = []
                        for fea_k in fea_k_list:
                            fea_ind = fea[:fea_k]
                            X_, y_ = X_train[:, fea_ind], y_train
                            X__, y__ = X_valid[:, fea_ind], y_valid
                            acc_ = cly_data(X_, y_, X__, y__, bcn)
                            accs.append(acc_)
                        fs_num, fs_acc = find_peak(accs, fea_step, fea_num, beta)
                        fs_nums.append(fs_num) 
                        fs_accs.append(fs_acc)
                    np.savetxt(fea_file, fs_nums, delimiter=',', fmt='%d')
                    clock_stop = time.time()
                    print('%s %s %d fs iteration cost %s seconds.' % (fn, mc, col_i, clock_stop-clock_start))

def feature_select(bcn, names, fs_tags, mat_codes, in_dir, mat_dir, spt_dir, fea_dir, out_dir, exp_num):
    """Form feature subsets"""
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    print("select feature...")
    for idx in range(exp_num):
        for fs in fs_tags:
            for dn in names:    
                for mc in mat_codes:
                    print(idx, " ", fs, " ", dn, " ", mc)
                    mat_file = '%s%s_%s_%d.csv' % (mat_dir, dn, mc, idx)
                    mat = np.loadtxt(mat_file, np.int, delimiter=',')   
                    mat_len = mat.shape[1]
                    for col_i in range(mat_len):
                        spt_f = '%s%s_%s_%s_%s_%d_%d.csv' % (spt_dir, bcn, dn, mc, fs, idx, col_i)
                        trn_i_f, trn_o_f = '%s%s_%d_train.csv' % (in_dir, dn, idx), '%s%s_%s_%s_%s_%d_%d_train.csv' % (out_dir, bcn, dn, mc, fs, idx, col_i)
                        vld_i_f, vld_o_f = '%s%s_%d_valid.csv' % (in_dir, dn, idx), '%s%s_%s_%s_%s_%d_%d_valid.csv' % (out_dir, bcn, dn, mc, fs, idx, col_i)
                        tst_i_f, tst_o_f = '%s%s_%d_test.csv' % (in_dir, dn, idx), '%s%s_%s_%s_%s_%d_%d_test.csv' % (out_dir, bcn, dn, mc, fs, idx, col_i)
                        
                        fea_file = '%s%s_%s_%s_%d_%d_fea_num.csv' % (fea_dir, bcn, dn, mc, idx, col_i)
                        fea_nums = np.loadtxt(fea_file, delimiter=',', dtype=np.int)
                        fea_num=fea_nums[fs_tags.index(fs)]
                        
                        fea_slt(trn_i_f, spt_f, trn_o_f, fea_num)
                        fea_slt(vld_i_f, spt_f, vld_o_f, fea_num)
                        fea_slt(tst_i_f, spt_f, tst_o_f, fea_num)
                        