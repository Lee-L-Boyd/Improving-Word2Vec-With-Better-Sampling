#!/usr/bin/env python
#import pip
#pip.main(['install','pandas'])
import sys
import pandas as pd
import json
import nltk
from guess_language import guess_language
from nltk.tokenize import word_tokenize
import unicodedata
import re
import pickle
window_size = 5
for line in sys.stdin:
  #df = pd.DataFrame.from_csv('top.csv').head(200000)
  #dictionary = dict()
  #for token in df['word']:
  #  dictionary[token] = token
  dictionary = pickle.load(open('200k_dictionary', 'r'))
  try:
    d=json.loads(line)
    line = d['text'].strip()
    if guess_language(line)!='en' or len(line) < window_size * 2:
      #print '%s\t%s\t%s\t%s' % ('NOTENGLISHORTOOSHORT','NA', 1, 1)
      pass
    else:
      words = line.lower().encode('ascii','ignore')
      words2 = word_tokenize(re.sub('\d', '0', words))
      #print(df.head(10))
      word_list = []
      #window_size = 2
      #para = ['this', 'is', 'a', 'test', 'for', ',', '\'', 'school', 'an', '.']
      for word in words2:
        try:
          dictionary[word]
          word_list.append(word)
          #if df['word'].str.contains('^'+str(re.escape(word))+'$').any():
          #print(str(word) + " found")
          #word_list.append(str(word))
        except:
	  pass
          #print '%s@#@%s\t%s\%s' % ('FAILEDFAILEDFAILED', word, 1, 1)
      #else:
      #  print(str(word) + " not in" + str(df.head(10)['word']))

      #print(para)
      para = word_list
      #print(para)
      #para = word_list
      tuples = []
      for index, word in enumerate(para):
        for j in range(window_size+1):
          if (index - j) >= 0 and j>0:
            print '%s@#@%s@#@%s\t%s' % (word, para[index-j], str(j), str(1))
          if (index + j + 1) <len(para) and j<window_size:
            print '%s@#@%s@#@%s\t%s' % (word, para[index + j + 1], str(j+1), str(1))
        print '%s@#@%s@#@%s\t%s' % (word, word, str(0), str(1))

        #for item in tuples:
        #  print '%s@#@%s@#@%s\t%s' % (item[0], item[1], item[2], item[3])
  except:
    print '%s@#@%s@#@%s\t%s' % ("FAILURESOMEWHERE", 'SOMEWHERE', 0, 0)
