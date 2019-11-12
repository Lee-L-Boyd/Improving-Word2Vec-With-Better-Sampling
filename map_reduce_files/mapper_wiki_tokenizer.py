#!/usr/bin/env python

import sys
import re 
from nltk.tokenize import StanfordTokenizer
reload(sys)
sys.setdefaultencoding('utf8')
from guess_language import guess_language

# input comes from STDIN (standard input)
def keepTheIndex(index, listOfRemovals):
	keepIt = True
	for removal in listOfRemovals:
		if index >= removal[0] and index <= removal[1]:
			keepIt = False
	return keepIt
for file in sys.stdin:
	x=1
	#f = open('wikipedia_dataset/wiki1-n-line.xml', 'r')
	#lines = f.read().encode('utf-8').split('</text>')
#for index, line in enumerate(lines):
    	# remove leading and trailing whitespace
	lines = file.encode('utf-8').split('</text>')
	for line in lines:
			newline = ''
    			try:
				if "<text xml:space=\"preserve\">" in line and "#REDIRECT" not in line:
					newline = line[line.find("<text xml:space=\"preserve\">")+len("<text xml:space=\"preserve\">"):]
					if guess_language(newline) == 'en':
						s=re.sub('[^A-Za-z0-9\s.,\'\";?$%+-:!]+', '@', re.sub('\d', '0', newline).replace('[', ' ').replace(']', ' ').replace('}', ' ').replace('{', ' '))
						s2 = StanfordTokenizer().tokenize(s)
						s3 = [word.encode('ascii') for word in s2 ]
						charCounter = 0;
						tokenCounter = 0;
						sentStart = 0;
						deleteThese = []
						for index, token in enumerate(s3):
							if token == '.':
								if charCounter < 20 or tokenCounter < 5:
									deleteThese.append([sentStart, index])
								charCounter = 0
								tokenCounter = 0
								sentStart = index + 1
							else:
								charCounter += len(token)
								tokenCounter += 1 
					        s3 = [ word for index, word in enumerate(s3) if keepTheIndex(index, deleteThese) ]
						print s3
			except:
				continue

