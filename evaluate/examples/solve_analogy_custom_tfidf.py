# -*- coding: utf-8 -*-

"""
 Simple example showing answering analogy questions
"""
import logging
from web.datasets.analogy import fetch_google_analogy
from web.embeddings import fetch_SG_GoogleNews, load_embedding

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

# Fetch skip-gram trained on GoogleNews corpus and clean it slightly
#w = fetch_SG_GoogleNews(lower=True, clean_words=True)
kargs = {'vocab_size':200000, 'dim':400}
fname='/home/student/Desktop/paper_1/hadoop-1.2.1/1_sparse_matrix/tfidf_nospan_embeddings2-2'
w = load_embedding(fname, format="glove", normalize=True,
                   lower=True, clean_words=False, load_kwargs=kargs)
# Fetch analogy dataset
data = fetch_google_analogy()

for cat in (set(data.category)):
    print(cat)

# Pick a sample of data and calculate answers
'''subset = [50, 1000, 4000, 10000, 14000]
for id in subset:
    w1, w2, w3 = data.X[id][0], data.X[id][1], data.X[id][2]
    print("Question: {} is to {} as {} is to ?".format(w1, w2, w3))
    print("Answer: " + data.y[id])
    print("Predicted: " + " ".join(w.nearest_neighbors(w[w2] - w[w1] + w[w3], exclude=[w1, w2, w3])))'''
score = 0.0
total = 0.0
for i, d in enumerate(data.X):
    try:
      w1, w2, w3 = d[0], d[1], d[2]
      if " ".join(w.nearest_neighbors(w[w2] - w[w1] + w[w3], exclude=[w1, w2, w3])) == data.y[i]:
        score += 1.0
      total += 1.0
    except:
      pass
    if i % 100 == 0:
      print(str(i) + ", " + str(score/total))
print(score/total)
