../bin/hadoop fs -rmr /user/mr/row_count
../bin/hadoop jar ../contrib/streaming/hadoop-*streaming*.jar \
 -D stream.num.map.output.key.fields=2 \
 -D mapred.text.key.comparator.options='-k1 -k2' \
 -D mapred.text.key.partitioner.options='-k1' \
 -D map.output.key.field.separator='@#@' \
 -D mapred.reduce.tasks=1 \
 -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
 -file ~/hadoop-1.2.1/run_mr/m_echo.py \
 -mapper ~/hadoop-1.2.1/run_mr/m_echo.py \
 -file ~/hadoop-1.2.1/run_mr/r_row_count.py \
 -reducer ~/hadoop-1.2.1/run_mr/r_row_count.py \
 -input /user/mr/context_count/part-* \
 -output /user/mr/row_count
rm ~/hadoop-1.2.1/cache_files/row_count_result
../bin/hadoop fs -copyToLocal /user/mr/row_count/part-00000 ~/hadoop-1.2.1/cache_files/row_count_result

python ~/hadoop-1.2.1/1_cache_files/make_row_count_object.py

python ~/hadoop-1.2.1/1_cache_files/compute_row_count_total.py

../bin/hadoop fs -copyFromLocal \
~/hadoop-1.2.1/1_cache_files/dictionary_count_tuple.pickle \
/cache/count_dictionary

../bin/hadoop fs -copyFromLocal \
~/hadoop-1.2.1/1_cache_files/rows_total.pickle \
/cache/total 
