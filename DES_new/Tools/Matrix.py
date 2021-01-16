import os
import numpy as np
from CodeMatrix.Classifier import OVA_ECOC, OVO_ECOC, Dense_random_ECOC, Sparse_random_ECOC, D_ECOC, DC_ECOC, ECOC_ONE
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
                    func = OVA_ECOC()
                elif mat == 'OVO':
                    func = OVO_ECOC()
                elif mat == 'DEcoc':
                    func = D_ECOC()
                elif mat == 'DenseRand':
                    func = Dense_random_ECOC()
                elif mat == 'SparseRand':
                    func = Sparse_random_ECOC()
                elif mat == 'DC_ECOC':
                    func = DC_ECOC()
                elif mat == 'ECOC_ONE':
                    func = ECOC_ONE()
                else:
                    print('Matrix code error!')

                i_f = '%s%s.csv' % (in_dir, dn)
                o_f = '%s%s_%s_%d.csv' % (out_dir, dn, mat, idx)
                if os.path.exists(o_f):
                    break
                if len(o_f) > 0:
                    print('creating %s matrix for %s' % (mat, dn))
                    get_mat(i_f, o_f, func, mat)
                    
def get_mat(i_f, o_f, func, mat):
    """Create coding matrix"""
    X, y = read_data(i_f)
    print(o_f)
    if isinstance(o_f, str):
        func.fit(X, y)
        m = func.matrix
        np.savetxt(o_f, m, fmt='%d', delimiter=',')
    elif isinstance(o_f, list):
        for oo in o_f:
            func.fit(X, y)
            m = func.matrix
            np.savetxt(oo, m, fmt='%d', delimiter=',')
    else:
        raise Exception('Unknown value of o_f')