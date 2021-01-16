import numpy as np
from itertools import combinations
from scipy.special import comb
from CodeMatrix.Matrix_tool import _exist_same_col, _exist_same_row, _exist_two_class, _get_data_subset, _estimate_weight
from CodeMatrix.SFFS import sffs
from CodeMatrix import Criterion
from CodeMatrix.Distance import euclidean_distance
import copy


def _matrix(X, y):
	""" This is an example function.

	Description:
		_matrix(X, y):
			Parameters:
				X: {array-like, sparse matrix}, shape = [n_samples, n_features]
					Training vector, where n_samples in the number of samples and n_features is the number of features.
				y: array-like, shape = [n_samples]
					Target vector relative to X
			Returns:
				M: 2-d array, shape = [n_classes, n_dichotomies]
					The coding matrix.
	"""
	pass


def ova(X, y):
	"""
		ONE-VERSUS-ONE ECOC
	"""
	index = {l: i for i, l in enumerate(np.unique(y))}
	matrix = np.eye(len(index)) * 2 - 1
	return matrix, index


def ovo(X, y):
	"""
		ONE-VERSUS-ONE ECOC
	"""

	index = {l: i for i, l in enumerate(np.unique(y))}
	groups = combinations(range(len(index)), 2)
	matrix_row = len(index)
	matrix_col = np.int(comb(len(index), 2))
	col_count = 0
	matrix = np.zeros((matrix_row, matrix_col))
	for group in groups:
		class_1_index = group[0]
		class_2_index = group[1]
		matrix[class_1_index, col_count] = 1
		matrix[class_2_index, col_count] = -1
		col_count += 1
	return matrix, index


def dense_rand(X, y):
	"""
		Dense random ECOC
	"""

	while True:
		index = {l: i for i, l in enumerate(np.unique(y))}
		matrix_row = len(index)
		if matrix_row > 3:
			matrix_col = np.int(np.floor(10 * np.log10(matrix_row)))
		else:
			matrix_col = matrix_row
		matrix = np.random.random((matrix_row, matrix_col))
		class_1_index = matrix > 0.5
		class_2_index = matrix < 0.5
		matrix[class_1_index] = 1
		matrix[class_2_index] = -1
		if (not _exist_same_col(matrix)) and (not _exist_same_row(matrix)) and _exist_two_class(matrix):
			return matrix, index
