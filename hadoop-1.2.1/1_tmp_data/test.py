boyband = open('boyband', 'r')
total = 0
for line in boyband:
  other, count = line.split('\t')
  total += int(count)
print(total)
context_count = open('context_count/part-00000', 'r')
my_set = {'the'}
for line in context_count:
  source, other  = line.split('@#@', 1)
  my_set.add(source)
print(len(my_set))
  
