../bin/hadoop fs -rmr /user/mr/context_count_preliminary_span
../bin/hadoop jar ../contrib/streaming/hadoop-*streaming*.jar \
 -D stream.num.map.output.key.fields=3 \
 -D mapred.text.key.comparator.options='-k1 -k2 -k3' \
 -D mapred.text.key.partitioner.options='-k1' \
 -D map.output.key.field.separator='@#@' \
 -D mapred.reduce.tasks=4 \
 -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
 -file ~/hadoop-1.2.1/1_run_mr/m_echo.py \
 -mapper ~/hadoop-1.2.1/1_run_mr/m_echo.py \
 -file ~/hadoop-1.2.1/1_run_mr/r_context_count_preliminary_span.py \
 -reducer ~/hadoop-1.2.1/1_run_mr/r_context_count_preliminary_span.py \
 -input /user/mr/get_contexts/*/part-00000 \
 -output /user/mr/context_count_preliminary_span

