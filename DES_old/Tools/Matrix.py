import os
import numpy as np
from CodeMatrix.Classifier import ova, ovo, dense_rand
from Tools.Read import read_data

def create_matrix(names, in_dir, out_dir, matrix_types, exp_num):
	print('create matrix...')
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	for idx in range(exp_num):
		for dn in names:
			for mat in matrix_types:

				func = None
				if mat == 'OVA':
					func = ova
				elif mat == 'OVO':
					func = ovo
				elif mat == 'DenseRand':
					func = dense_rand
				else:
					print('Matrix code error!')

				i_f = '%s%s.csv' % (in_dir, dn)
				o_f = '%s%s_%s_%d.csv' % (out_dir, dn, mat, idx)
				if os.path.exists(o_f):
					break
				if len(o_f) > 0:
					print('creating %s matrix for %s' % (mat, dn))
					get_mat(i_f, o_f, func)
                    
def get_mat(i_f, o_f, func):
	"""Create coding matrix"""
	X, y = read_data(i_f)
	if isinstance(o_f, str):
		m, i = func(X, y)
		np.savetxt(o_f, m, fmt='%d', delimiter=',')
	elif isinstance(o_f, list):
		for oo in o_f:
			m, i = func(X, y)
			np.savetxt(oo, m, fmt='%d', delimiter=',')
	else:
		raise Exception('Unknown value of o_f')
