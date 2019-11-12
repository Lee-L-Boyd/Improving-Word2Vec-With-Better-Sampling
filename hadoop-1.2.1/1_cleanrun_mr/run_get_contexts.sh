
for i in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 
do 
../bin/hadoop fs -rmr /user/mr/get_contexts/${i}
../bin/hadoop jar ../contrib/streaming/hadoop-*streaming*.jar \
 -D stream.num.map.output.key.fields=3 \
 -D mapred.text.key.comparator.options='-k1 -k2 -k3' \
 -D mapred.text.key.partitioner.options='-k1' \
 -D map.output.key.field.separator='@#@' \
 -D mapred.reduce.tasks=1 \
 -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
 -file ~/hadoop-1.2.1/1_run_mr/m_get_contexts.py \
 -mapper ~/hadoop-1.2.1/1_run_mr/m_get_contexts.py \
 -file ~/hadoop-1.2.1/1_run_mr/m_get_contexts.py \
 -reducer ~/hadoop-1.2.1/1_run_mr/r_get_contexts.py \
 -file ~/hadoop-1.2.1/1_run_mr/r_get_contexts.py \
 -combiner ~/hadoop-1.2.1/1_run_mr/r_get_contexts.py \
 -cacheFile /cache/200k_dictionary#200k_dictionary \
 -input /user/mr/corpus/j${i}/AA/wiki_* \
 -output /user/mr/get_contexts/${i}

done
