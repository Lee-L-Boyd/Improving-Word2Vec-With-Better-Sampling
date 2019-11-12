import pickle
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from numpy.random import rand
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds, eigs
matrix1 = pickle.load(open('pmi_nospan_matrix1','r'))
matrix2 = pickle.load(open('pmi_nospan_matrix2', 'r'))
matrix3 = pickle.load(open('pmi_nospan_matrix3','r'))
matrix4 = pickle.load(open('pmi_nospan_matrix4', 'r'))
matrix5 = pickle.load(open('pmi_nospan_matrix_final', 'r'))
matrix6 = matrix1 + matrix2 + matrix3 + matrix4 + matrix5
#print(matrix6)
u, s, vt = svds(matrix6, k=400)
#print("found pieces")
print(u.dot(s).shape)
reduced_matrix = matrix6.dot(vt.T)
pickle.dump(reduced_matrix, open('pmi_nospan_reduced', 'wb'))
