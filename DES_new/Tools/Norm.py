import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Tools.Read import read_data

"""
Norm data

Modify:
    2021-1-16
"""
def normalize_data(names, in_dir, out_dir, exp_num):
	print('normalize data...')
	suffix = ['_train.csv', '_valid.csv', '_test.csv']
	nor_obj = MinMaxScaler()

	for idx in range(exp_num):
		for dn in names:
			for suf in suffix:
				i_f, o_f = '%s%s_%d%s' % (in_dir, dn, idx, suf), '%s%s_%d%s' % (out_dir, dn, idx, suf)
				norm(nor_obj, i_f, o_f)
				
def norm(norm_obj, i_f, o_f):
	X, y = read_data(i_f)  
	X_new = norm_obj.fit_transform(X, y)
	X_new = np.c_[X_new, y]
	np.savetxt(o_f, X_new, fmt='%s', delimiter=',')