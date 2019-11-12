# -*- coding: utf-8 -*-

"""
 Simple example showing evaluating embedding on similarity datasets
"""
import logging
from six import iteritems
from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999
from web.embeddings import fetch_GloVe, load_embedding
from web.evaluate import evaluate_similarity
import sys

#first argument is count method (sys.argv[1])
#second argument is weight method (sys.argv[2])

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

# Fetch GloVe embedding (warning: it might take few minutes)
#w_glove = fetch_GloVe(corpus="wiki-6B", dim=300)
kargs = {'vocab_size':200000, 'dim':400}
fname='/home/student/Desktop/paper_1/hadoop-1.2.1/1_sparse_matrix/'+str(sys.argv[2])+'_'+str(sys.argv[1])+'_'+'embeddings'
w_custom = load_embedding(fname, format="glove", normalize=True,
                   lower=True, clean_words=False, load_kwargs=kargs)
# Define tasks
tasks = {
    "MEN": fetch_MEN(),
    "WS353": fetch_WS353(),
    "SIMLEX999": fetch_SimLex999()
}

# Print sample data
for name, data in iteritems(tasks):
    print("Sample data from {}: pair \"{}\" and \"{}\" is assigned score {}".format(name, data.X[0][0], data.X[0][1], data.y[0]))

# Calculate results using helper function
for name, data in iteritems(tasks):
    print("Spearman correlation of scores on {} {}".format(name, evaluate_similarity(w_custom, data.X, data.y)))
