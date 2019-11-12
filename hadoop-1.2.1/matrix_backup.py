import sys
import pickle
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from numpy.random import rand
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds, eigs

from multiprocessing import Pool
from functools import reduce
sys.path.insert(0, '/home/rcx653/1_paper/evaluate/examples')
import evaluate_similarity_module

#arg 1 should be count method
#arg 2 should be weight method
def getVals(d):
  row = []
  col = []
  val = []
  for line in d:
    line = line.strip()
    try:
      line = line.replace('@#@', '\t', 1)
      vals = line.split('\t', 2)
      vals[0] = int(vals[0])
      vals[1] = int(vals[1])
      vals[2] = float(vals[2])
      row.append(vals[0])
      col.append(vals[1])
      val.append(vals[2])
    except: 
      pass
  return csr_matrix((val, (row, col)), shape=(200000, 200000))
filename = str(sys.argv[2]) + '_' + str(sys.argv[1])
automate = False
if input('Would you like to automate the whole process? (y) or (n) ') == 'y':
  automate = True
if automate or input('Would you like to convert to matrix? (y) or (n) ') == 'y':
  previous_prompt = True
  f = open('pmi_results/' + filename)
  p = Pool(35)
  file_list = list(f)
  length = len(file_list)
  splits = length//34
  pool_list=[]
  chunks = [file_list[x:x+splits] for x in range(0, length, splits)]
  data = [x for x in p.map(getVals, chunks) if x is not None]
  matrix = reduce(lambda x,y: x+y, data)
  p.close()
  matrix_name = filename+'_matrix'
  final = open(matrix_name, 'wb')
  pickle.dump(matrix, final)
  f.close()
  data = []
else:
  previous_prompt = False
if automate or input('Would you like to reduce? (y) or (n) ') == 'y':
  if not previous_prompt and not automate:
    matrix_file = filename+'_matrix'
    matrix = pickle.load(open(matrix_file, 'rb'))
  previous_prompt == True
  print("Reducing now")
  matrix = matrix.asfptype()
  u, s, vt = svds(matrix, k=400)
  reduced_matrix = np.dot(u, np.sqrt(np.diag(s)))
  pickle.dump(reduced_matrix, open(filename+'_reduced', 'wb'))
  matrix = None
else:
  previous_prompt = False
out_filename = filename+'_embeddings'
if automate or input('Would you like to change to glove format (y) or (n) ') == 'y':
  if not previous_prompt and not automate:
    reduced_file = filename+'_reduced'
    reduced_matrix = pickle.load(open(reduced_file, 'rb'))
  previous_prompt == True
  print("Writing to glove format")
  r_dictionary = pickle.load(open('../1_cache_files/200k_rdictionary', 'rb'))
  f = open(out_filename, 'w')
  for i, v in enumerate(reduced_matrix):
    f.write(r_dictionary[i])
    for value in v:
      if value == 0:
        f.write(' ' + str(.0000001))
      else:
        f.write(' ' + str(value+.0000001))
    try:
      r_dictionary[i+1]
      f.write('\n')
    except:
      pass
  f.close()
else:
  previous_prompt = False
if automate or input('Would you like to evaluate (y) or (n) ') == 'y':
  evaluate_similarity_module.call_module(out_filename)
