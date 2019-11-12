# -*- coding: utf-8 -*-

"""
 Simple example showing answering analogy questions
"""
import logging
from web.datasets.analogy import fetch_google_analogy
from web.embeddings import fetch_SG_GoogleNews
from web.embeddings import fetch_GloVe
# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

# Fetch skip-gram trained on GoogleNews corpus and clean it slightly
#w = fetch_SG_GoogleNews(lower=True, clean_words=True)
w = fetch_GloVe(corpus="wiki-6B", dim=300)
# Fetch analogy dataset
data = fetch_google_analogy()

for cat in (set(data.category)):
    print(cat)

# Pick a sample of data and calculate answers
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

