import pickle
import pandas as pd


row_count=open("/home/mr/hadoop-1.2.1/1_cache_files/row_count_nospan_result","r")
#dict_file=open("out.csv","r")
#df = pd.DataFrame.from_csv('/home/mr/hadoop-1.2.1/1_cache_files/out.csv')
#dictionary = {}
#for index, token in enumerate(df['word']):
#  dictionary[token] = index
dictionary = pickle.load(open('/home/mr/hadoop-1.2.1/1_cache_files/200k_dictionary', 'r'))
dictionary_counts = {}
for line in row_count:
  try:
    line = line.strip()
    line = line.split('\t')
    dictionary_counts[line[0]] = int(line[1])
  except:
    pass
pickle.dump(dictionary_counts, open('/home/mr/hadoop-1.2.1/1_cache_files/row_count_nospan.pickle', 'w'))
row_count.close()
print(dictionary_counts)
