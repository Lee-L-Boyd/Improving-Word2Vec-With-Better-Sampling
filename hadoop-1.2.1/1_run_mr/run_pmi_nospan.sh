 ../bin/hadoop fs -rmr /user/mr/pmi_nospan
 ../bin/hadoop jar ../contrib/streaming/hadoop-*streaming*.jar \
  -D stream.num.map.output.key.fields=3 \
  -D mapred.text.key.comparator.options='-k1 -k2 -k3' \
  -D mapred.text.key.partitioner.options='-k1' \
  -D map.output.key.field.separator='@#@' \
  -D mapred.reduce.tasks=1 \
  -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
  -file ~/hadoop-1.2.1/1_run_mr/m_echo.py \
  -reducer ~/hadoop-1.2.1/1_run_mr/m_echo.py \
  -file ~/hadoop-1.2.1/1_run_mr/r_pmi_nospan.py \
  -mapper ~/hadoop-1.2.1/1_run_mr/r_pmi_nospan.py \
  -cacheFile /cache/200k_dictionary#200k_dictionary\
  -cacheFile /cache/row_count_nospan.pickle#row_count_nospan.pickle \
  -cacheFile /cache/row_count_total_nospan.pickle#row_count_total_nospan.pickle \
  -input /user/mr/context_count_nospan/part-* \
  -output /user/mr/pmi_nospan
rm ~/hadoop-1.2.1/1_sparse_matrix/pmi_results/pmi_nospan
../bin/hadoop fs -copyToLocal /user/mr/pmi_nospan/part-00000 ~/hadoop-1.2.1/1_sparse_matrix/pmi_results/pmi_nospan
