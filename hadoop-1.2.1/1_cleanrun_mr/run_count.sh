# first argument is span or nospan or none
# second argument row or context
# third argument is binary or regular
# NOTE: there is no binary span combo
rm count_parameters.txt
echo "$1" >> count_parameters.txt
echo "$2" >> count_parameters.txt
echo "$3" >> count_parameters.txt
  ~/hadoop-1.2.1/bin/hadoop fs -rmr /params
  ~/hadoop-1.2.1/bin/hadoop fs -copyFromLocal ~/hadoop-1.2.1/1_cleanrun_mr/parameters.txt /params/parameters.txt
  ../bin/hadoop fs -rmr /user/mr/context_count_span
  ../bin/hadoop jar ../contrib/streaming/hadoop-*streaming*.jar \
  -D stream.num.map.output.key.fields=2 \
  if ["$2" = "context"]
    -D mapred.text.key.comparator.options='-k1 -k2 -k3' \
  else
    -D mapred.text.key.comparator.options='-k1 -k2' \
  fi
  -D mapred.text.key.partitioner.options='-k1' \
  -D map.output.key.field.separator='@#@' \
  if ["$2" = "context"]
  then
    -D mapred.reduce.tasks=4 \
  else
    -D mapred.reduce.tasks=1 \
  fi
  -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
  -file ~/hadoop-1.2.1/1_run_mr/m_echo.py \
  -mapper ~/hadoop-1.2.1/1_run_mr/m_echo.py \
  -file ~/hadoop-1.2.1/1_run_mr/r_$2_count_$1.py \
  -reducer ~/hadoop-1.2.1/1_run_mr/r_$2_count_$1.py \
  if ["$2" = "context"]
  then
    -input /user/mr/get_contexts/*/part-00000 \
  elif ["$1" = "binary"]
  then
    -input /user/mr/context_count_preliminary_span/part-* \
  -output /user/mr/context_count_span

