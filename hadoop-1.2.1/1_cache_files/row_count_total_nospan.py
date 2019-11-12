import pickle
row_count = open("/home/mr/hadoop-1.2.1/1_cache_files/row_count_nospan_result", "r")
total = 0
for line in row_count:
  line = line.strip()
  line = line.split('\t')
  total+=int(line[1])
pickle.dump(str(total), open('/home/mr/hadoop-1.2.1/1_cache_files/row_count_total_nospan.pickle','w'))
print total
row_count.close()
