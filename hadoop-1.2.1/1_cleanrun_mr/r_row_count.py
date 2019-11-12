#!/usr/bin/env python
"""reducer.py"""

from operator import itemgetter
import sys

current_source = None
current_context = None
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
      count = int(count)
      #span = int(span)
    except ValueError:
        # count was not a number, so silently
        # ignore/discard this line
        continue

    # this IF-switch only works because Hadoop sorts map output
    # by key (here: word) before it is passed to the reducer
    if current_source == source:
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
if current_source == source:
    print '%s\t%s' % (current_source, current_count)
