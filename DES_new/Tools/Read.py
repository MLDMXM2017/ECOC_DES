import numpy as np
	
def read_data(fil):
    X = np.loadtxt(fil, delimiter=',', dtype=np.str)
    y = X[:, -1]
    X = X[:, :-1].astype(np.float)
    return X, y

def read_microarray(fil):
    X = np.loadtxt(fil, delimiter=',', dtype=np.str)
    y = X[0, :]
    X = X[1:, :].T.astype(np.float)
    return X, y