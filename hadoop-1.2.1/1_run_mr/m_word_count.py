#!/usr/bin/env python
import sys
import pandas as pd
import json
import nltk
from guess_language import guess_language
from nltk.tokenize import word_tokenize
import unicodedata
import re
window_size = 5
for line in sys.stdin:
    try:
      d=json.loads(line)
      line = d['text'].strip()
      if guess_language(line)!='en' or len(line) < window_size * 2:
        pass
      else:
        words = line.lower().encode('ascii','ignore')
        words2 = word_tokenize(re.sub('\d', '0', words))
        for word in words2:
          try:
            print('%s\t%s'%(word, 1))
          except:
            pass
    except:
      pass
