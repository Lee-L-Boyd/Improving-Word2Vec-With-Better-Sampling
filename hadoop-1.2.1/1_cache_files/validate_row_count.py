import pickle
test_rc = pickle.load(open('row_count_nospan.pickle','r'))
total = pickle.load(open('row_count_total_nospan.pickle','r'))
print(len(test_rc.keys()))
#print(test_object['the'][1])
print(total)
