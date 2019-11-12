#!/usr/bin/env python
"""reducer.py"""

from operator import itemgetter
import sys
import pickle
import math
import sys
import time
#first argument is the count type 
#second argument is the weight type

f = open('parameters.txt', 'r')
for i, arg in enumerate(f):
  if i == 0:
    count_type = str(arg.strip())
  else:
    weight_type = str(arg.strip())
f.close()

#count_type = "span"
#weight_type = "pmi"
#row_count_file = 'row_count_'+count_type+'.pickle'
#row_count_total_file = 'row_count_total_'+count_type+'.pickle'
#binary_row_count_file = 'binary_row_count_'+count_type +'.pickle'
row_count = pickle.load(open('row_count.pickle', 'r'))
row_count_total = pickle.load(open('row_count_total.pickle', 'r'))
binary_row_count = pickle.load(open('binary_row_count.pickle', 'r'))
#time.sleep(2)
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
def pmi_function():
  #raise NameError('THE RESULTS ARE NOT IN%s!!' % (source +', '+ context))
  cell_value = math.log((count/float(row_count[source]))*(row_count_total/float(row_count[context])))
  if cell_value > -100:
    #raise NameError('THE RESULTS ARE IN%s!!' % (source +', '+ context))
    print('%s@#@%s\t%s' % (str(new_source), str(new_context), str(cell_value)))
  #return cell_value

def tfidf_function():
  new_count = math.log((count/float(row_count[source]))*(200000/float(binary_row_count[context]+1)))
  if new_count > -100:
    #raise NameError('THE RESULTS ARE IN%s!!' % (source +', '+ context))
    print('%s@#@%s\t%s' % (str(new_source), str(new_context), str(new_count)))

def sppmi1_span_function():
  #raise NameError('PPMI')
  new_count = math.log((count/float(row_count[source]))*(row_count_total/float(row_count[context])))-math.log(1.0/3.0)
  if new_count > 0:
    print('%s@#@%s\t%s' % (str(new_source), str(new_context), str(new_count)))

def sppmi3_span_function():
  new_count = math.log((count/float(row_count[source]))*(row_count_total/float(row_count[context])))-math.log(1)
  if new_count > 0:
    print('%s@#@%s\t%s' % (str(new_source), str(new_context), str(new_count)))


def sppmi5_span_function():
  new_count = math.log((count/float(row_count[source]))*(row_count_total/float(row_count[context])))-math.log(5.0/3.0)
  if new_count > 0:
    print('%s@#@%s\t%s' % (str(new_source), str(new_context), str(new_count)))

def sppmi15_span_function():
  new_count = math.log((count/float(row_count[source]))*(row_count_total/float(row_count[context])))-math.log(15.0/3.0)
  if new_count > 0:
    print('%s@#@%s\t%s' % (str(new_source), str(new_context), str(new_count)))

def sppmi1_nospan_function():
  #raise NameError('PPMI')
  new_count = math.log((count/float(row_count[source]))*(row_count_total/float(row_count[context])))-math.log(1.0)
  if new_count > 0:
    print('%s@#@%s\t%s' % (str(new_source), str(new_context), str(new_count)))

def sppmi5_nospan_function():
  new_count = math.log((count/float(row_count[source]))*(row_count_total/float(row_count[context])))-math.log(5.0)
  if new_count > 0:
    print('%s@#@%s\t%s' % (str(new_source), str(new_context), str(new_count)))

def sppmi15_nospan_function():
  new_count = math.log((count/float(row_count[source]))*(row_count_total/float(row_count[context])))-math.log(15.0)
  if new_count > 0:
    print('%s@#@%s\t%s' % (str(new_source), str(new_context), str(new_count)))

def sppmi5tfidf_nospan_function():
  df = count/float(row_count[source])
  idf = math.log(200000/float(binary_row_count[context]+1))
  inversepcontext = float(row_count_total)/float(row_count[context])
  new_count = math.log(df*inversepcontext*idf) - math.log(idf*5)
  if new_count > 0:
    print('%s@#@%s\t%s' % (str(new_source), str(new_context), str(new_count)))

def sppmi15tfidf_nospan_function():
  df = count/float(row_count[source])
  idf = math.log(200000/float(binary_row_count[context]+1))
  inversepcontext = float(row_count_total)/float(row_count[context])
  new_count = math.log(df*inversepcontext*idf) - math.log(idf*15)
  if new_count > 0:
    print('%s@#@%s\t%s' % (str(new_source), str(new_context), str(new_count)))

def sppmi5tfidf_span_function():
  df = count/float(row_count[source])
  idf = math.log(200000/float(binary_row_count[context]+1))
  inversepcontext = float(row_count_total)/float(row_count[context])
  new_count = math.log(df*inversepcontext*idf) - math.log(idf*15/3)
  if new_count > 0:
    print('%s@#@%s\t%s' % (str(new_source), str(new_context), str(new_count)))


weight_dict = {
  'pmispan':pmi_function,
  'tfidfspan':tfidf_function,
  'pminospan':pmi_function,
  'tfidfnospan':tfidf_function,
  'sppmi1span':sppmi1_span_function,
  'sppmi5span':sppmi5_span_function,
  'sppmi15span':sppmi15_span_function,
  'sppmi1nospan':sppmi1_nospan_function,
  'sppmi5nospan':sppmi5_nospan_function,
  'sppmi15nospan':sppmi15_nospan_function,
  'sppmi3span':sppmi3_span_function,
  'sppmi5tfidfnospan':sppmi5tfidf_nospan_function,
  'sppmi15tfidfnospan':sppmi15tfidf_nospan_function,
  'sppmi5tfidfspan':sppmi5tfidf_span_function,
}

verification_dictionary = {}
for line in sys.stdin:
  


  success = True
  line = line.strip()
  #try:
  if True:
    line = line.replace('@#@', '\t', 1)
    source, context, count = line.split('\t', 2)
    count = float(count)
    if count > 0:
      new_source = dictionary[source]
      new_context = dictionary[context]
      weight_dict[str(weight_type+count_type)]()
  #except:
  else:
    success = False
    pass
'''  try:
    verification_dictionary[source+context]
    verification_dictionary[source+context] = False 
  except:
    verification_dictionary[source+context] = True
  if success and not verification_dictionary[source+context]:
    raise NameError('The map file is not fully reduced which will give inaccurate results %s!!' % (source +', '+ context))
'''
