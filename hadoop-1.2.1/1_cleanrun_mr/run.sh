# first variable is the count type
# second variable is the weight type
#rm mr.sh
rm parameters.txt
echo "$1" >>parameters.txt
echo "$2" >>parameters.txt
~/hadoop-1.2.1/bin/hadoop fs -rmr /params
~/hadoop-1.2.1/bin/hadoop fs -copyFromLocal ~/hadoop-1.2.1/1_cleanrun_mr/parameters.txt /params/parameters.txt
#rm mr.sh
 ../bin/hadoop fs -rmr /user/mr/$2_$1
 ../bin/hadoop jar ../contrib/streaming/hadoop-*streaming*.jar \
  -D stream.num.map.output.key.fields=3 \
  -D mapred.text.key.comparator.options='-k1 -k2 -k3' \
  -D mapred.text.key.partitioner.options='-k1' \
  -D map.output.key.field.separator='@#@' \
  -D mapred.reduce.tasks=1 \
  -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
  -file ~/hadoop-1.2.1/1_run_mr/m_echo.py \
  -reducer ~/hadoop-1.2.1/1_run_mr/m_echo.py \
  -file ~/hadoop-1.2.1/1_cleanrun_mr/r.py \
  -mapper ~/hadoop-1.2.1/1_cleanrun_mr/r.py \
  -cacheFile /params/parameters.txt#parameters.txt \
  -cacheFile /cache/binary_row_count_$1.pickle#binary_row_count.pickle \
  -cacheFile /cache/200k_dictionary#200k_dictionary\
  -cacheFile /cache/row_count_$1.pickle#row_count.pickle \
  -cacheFile /cache/row_count_total_$1.pickle#row_count_total.pickle \
  -input /user/mr/context_count_$1/part-* \
  -output /user/mr/$2_$1

rm ~/hadoop-1.2.1/1_sparse_matrix/pmi_results/$2_$1
~/hadoop-1.2.1/bin/hadoop fs -copyToLocal /user/mr/$2_$1/part-00000 ~/hadoop-1.2.1/1_sparse_matrix/pmi_results/$2_$1
