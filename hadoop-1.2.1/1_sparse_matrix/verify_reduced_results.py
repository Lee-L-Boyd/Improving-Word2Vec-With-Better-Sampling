import pickle
from sklearn.neighbors import NearestNeighbors
import numpy as np


rr = pickle.load(open('pmi_nospan_reduced', 'r'))
rr_np = np.array(rr)
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(rr_np)
distances, indices = nbrs.kneighbors(rr_np)
print(indices)
print(distances)
