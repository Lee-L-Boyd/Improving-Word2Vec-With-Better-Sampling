#!/usr/bin/env python

import sys
import re 

p = re.compile('\[*\'(.*)\'')
q = re.compile('\[*\"(.*)\"')
# input comes from STDIN (standard input)
#f = open('clean/wikiClean9-12/part-00000', 'r')
#for line in f:
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    # split the line into words
    words = line.split()
    # increase counters
    for word in words:
        # write the results to STDOUT (standard output);
        # what we output here will be the input for the
        # Reduce step, i.e. the input for reducer.py
        #
        # tab-delimited; the trivial word count is 1
        m = p.match(word.lower())
	n = q.match(word.lower())
	if m:
		#continue
		print '%s\t%s' % (m.group(1), 1)
	elif n:
		#continue
		print '%s\t%s' % (n.group(1), 1)
	else:
		print '%s\t%s' % ("NOMATCH " + str(word), 1)
