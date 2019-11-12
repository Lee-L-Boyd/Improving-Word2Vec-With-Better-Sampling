../bin/hadoop fs -rmr /user/mr/context_count_nospan
../bin/hadoop jar ../contrib/streaming/hadoop-*streaming*.jar \
 -D stream.num.map.output.key.fields=3 \
 -D mapred.text.key.comparator.options='-k1 -k2 -k3' \
 -D mapred.text.key.partitioner.options='-k1' \
 -D map.output.key.field.separator='@#@' \
 -D mapred.reduce.tasks=1 \
 -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
 -file ~/hadoop-1.2.1/1_run_mr/m_echo.py \
 -mapper ~/hadoop-1.2.1/1_run_mr/m_echo.py \
 -file ~/hadoop-1.2.1/1_run_mr/r_context_count_nospan.py \
 -reducer ~/hadoop-1.2.1/1_run_mr/r_context_count_nospan.py \
 -input /user/mr/get_contexts/*/part-00000 \
 -output /user/mr/context_count_nospan

