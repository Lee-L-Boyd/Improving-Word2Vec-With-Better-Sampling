import pickle
words = open('/home/mr/hadoop-1.2.1/1_cache_files/word_count', 'r')
word_list = []
for line in words:
  word, count = line.split('\t')
  count = int(count)
  #if count > 100:
  word_list.append((word,count))
word_list.sort(key = lambda tup: tup[1], reverse = True)
dictionary = {}
r_dictionary = {}
word_list = word_list[:200000]
for index, value in enumerate(word_list):
  dictionary[value[0]] = index
  r_dictionary[index] = value[0]
print(dictionary)
print(r_dictionary)
pickle.dump(dictionary, open('/home/mr/hadoop-1.2.1/1_cache_files/200k_dictionary', 'w'))
pickle.dump(r_dictionary, open('/home/mr/hadoop-1.2.1/1_cache_files/200k_rdictionary', 'w'))
