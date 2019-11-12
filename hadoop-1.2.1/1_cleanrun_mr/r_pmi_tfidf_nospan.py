#!/usr/bin/env python
"""reducer.py"""

from operator import itemgetter
import sys
import pickle
import math

row_count = pickle.load(open('row_count_nospan.pickle', 'r'))
row_count_total = pickle.load(open('row_count_total_nospan.pickle', 'r'))
binary_row_count = pickle.load(open('binary_row_count_nospan.pickle', 'r'))

dictionary = pickle.load(open('200k_dictionary', 'r'))
row_count_total = float(row_count_total)

current_source = None
current_context = None
current_span = None
current_count = 0
source = None
context = None
span = None
# input comes from STDIN
for line in sys.stdin:
    #print '%s\t%s' % (line)
    # remove leading and trailing whitespace
    line = line.strip()

    # parse the input we got from mapper.py
    #source, context, count  = line.split('\t', 2)

    # convert count (currently a string) to int
    try:
      line = line.replace('@#@', '\t', 1)
      source, context, count = line.split('\t', 2)
      #source, context = line.split('@#@', 1)  
      count = float(count)
      #span = int(span)
    except ValueError:
        # count was not a number, so silently
        # ignore/discard this line
        continue
    try:
      if count > 0:
        new_count = math.log((count/float(row_count[source]))*(float(row_count_total)/float(row_count[context]))*math.log(200000/float(binary_row_count[context]+1)))
        new_source = dictionary[source]
        new_context = dictionary[context]
        if new_count > -100:
          print('%s@#@%s\t%s' % (str(new_source), str(new_context), str(new_count)))
    except:
      pass
    # this IF-switch only works because Hadoop sorts map output
    # by key (here: word) before it is passed to the reducer
    '''if current_source == source:
        current_count += count
    else:
        if current_source:
            # write result to STDOUT
            print '%s\t%s' % (current_source, current_count)
        current_count = count
        current_source = source
        #current_context = context
        #current_span = span
# do not forget to output the last word if needed!
if current_source == source and current_context == context:
    print '%s\t%s' % (current_source, current_count)
'''
