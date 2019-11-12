#../bin/hadoop fs -rmr /user/mr/row_count_nospan
#../bin/hadoop jar ../contrib/streaming/hadoop-*streaming*.jar \
# -D stream.num.map.output.key.fields=2 \
# -D mapred.text.key.comparator.options='-k1 -k2' \
# -D mapred.text.key.partitioner.options='-k1' \
# -D map.output.key.field.separator='@#@' \
# -D mapred.reduce.tasks=1 \
# -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
# -file ~/hadoop-1.2.1/1_run_mr/m_echo.py \
# -mapper ~/hadoop-1.2.1/1_run_mr/m_echo.py \
# -file ~/hadoop-1.2.1/1_run_mr/r_row_count.py \
# -reducer ~/hadoop-1.2.1/1_run_mr/r_row_count.py \
# -input /user/mr/context_count_nospan/part-* \
# -output /user/mr/row_count_nospan
#rm ~/hadoop-1.2.1/1_cache_files/row_count_nospan_result
#../bin/hadoop fs -copyToLocal /user/mr/row_count_nospan/part-00000 ~/hadoop-1.2.1/1_cache_files/row_count_nospan_result

python ~/hadoop-1.2.1/1_cache_files/row_count_nospan.py

python ~/hadoop-1.2.1/1_cache_files/row_count_total_nospan.py

../bin/hadoop fs -rm /cache/row_count_nospan.pickle
../bin/hadoop fs -copyFromLocal \
~/hadoop-1.2.1/1_cache_files/row_count_nospan.pickle \
/cache/row_count_nospan.pickle

../bin/hadoop fs -rm /cache/row_count_total_nospan.pickle

../bin/hadoop fs -copyFromLocal \
~/hadoop-1.2.1/1_cache_files/row_count_total_nospan.pickle \
/cache/row_count_total_nospan.pickle
