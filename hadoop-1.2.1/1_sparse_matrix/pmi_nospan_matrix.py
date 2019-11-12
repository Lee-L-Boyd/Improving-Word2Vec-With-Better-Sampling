#import pip
#pip.main(['install','scipy'])
import pickle
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from numpy.random import rand
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds, eigs

x = 1
rows = []
cols = []
vals = []
with open('/home/mr/hadoop-1.2.1/1_sparse_matrix/pmi_results/pmi_nospan') as f:
  for line in f:
    #print(x)
    #print(line)
    line = line.strip()
    try:
      line = line.replace('@#@', '\t', 1)
      source, context, value = line.split('\t', 2)
    except:
      if len(line) > 2:
        print("unable to split line " + str(x) + " "  + line)
    try:
      rows.append(int(source))
      cols.append(int(context))
      vals.append(float(value))
    except:
      if len(line) > 2:
        print("unable to append line " + str(x) + " " + line) 
    x += 1
    if x % 100000000 == 0:
      print(x) 

      matrix = csr_matrix((vals, (rows, cols)), shape=(200000, 200000))
      #print(matrix)
      matrix_name = 'pmi_nospan_matrix' + str(x/100000000)
      final = open(matrix_name, 'w')
      pickle.dump(matrix, final)
      final.close()
      rows = []
      cols = []
      vals = []
matrix = csr_matrix((vals, (rows, cols)), shape=(200000, 200000))
matrix_name = 'pmi_nospan_matrix_final'
final = open(matrix_name, 'w')
pickle.dump(matrix, final)
final.close()
print(matrix)
f.close()

#u, s, vt = svds(matrix, k=2)
#print("found pieces")
#print(u.dot(s).shape)
#print(matrix.dot(vt.T)[199761])
