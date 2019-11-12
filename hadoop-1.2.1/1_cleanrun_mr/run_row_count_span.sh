../bin/hadoop fs -rmr /user/mr/row_count_span
../bin/hadoop jar ../contrib/streaming/hadoop-*streaming*.jar \
 -D stream.num.map.output.key.fields=2 \
 -D mapred.text.key.comparator.options='-k1 -k2' \
 -D mapred.text.key.partitioner.options='-k1' \
 -D map.output.key.field.separator='@#@' \
 -D mapred.reduce.tasks=1 \
 -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
 -file ~/hadoop-1.2.1/1_run_mr/m_echo.py \
 -mapper ~/hadoop-1.2.1/1_run_mr/m_echo.py \
 -file ~/hadoop-1.2.1/1_run_mr/r_row_count_span.py \
 -reducer ~/hadoop-1.2.1/1_run_mr/r_row_count_span.py \
 -input /user/mr/context_count_final_span/part-* \
 -output /user/mr/row_count_span
rm ~/hadoop-1.2.1/1_cache_files/row_count_span_result
../bin/hadoop fs -copyToLocal /user/mr/row_count_span/part-00000 ~/hadoop-1.2.1/1_cache_files/row_count_span_result

python ~/hadoop-1.2.1/1_cache_files/row_count_span.py

python ~/hadoop-1.2.1/1_cache_files/row_count_total_span.py

../bin/hadoop fs -rm /cache/row_count_span.pickle
../bin/hadoop fs -copyFromLocal \
~/hadoop-1.2.1/1_cache_files/row_count_span.pickle \
/cache/row_count_span.pickle

../bin/hadoop fs -rm /cache/row_count_total_span.pickle

../bin/hadoop fs -copyFromLocal \
~/hadoop-1.2.1/1_cache_files/row_count_total_span.pickle \
/cache/row_count_total_span.pickle
