bin/hadoop jar contrib/streaming/hadoop-*streaming*.jar \
 -D stream.num.map.output.key.fields=3 \
 -D mapred.text.key.comparator.options='-k1 -k2 -k3' \
 -D mapred.text.key.partitioner.options='-k1' \
 -D map.output.key.field.separator='@#@' \
 -D mapred.reduce.tasks=1 \
 -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
 -file ~/hadoop-1.2.1/reducer_echo_v2.py \
 -mapper ~/hadoop-1.2.1/reducer_echo_v2.py \
 -file ~/hadoop-1.2.1/reducer_count_final.py \
 -reducer ~/hadoop-1.2.1/reducer_count_final.py \
 -file ~/hadoop-1.2.1/reducer_count_final.py \
 -combiner ~/hadoop-1.2.1/reducer_count_final.py \
 -cacheFile /cache/top2.csv#top.csv \
 -input /user/mr/final_multi \
 -output /user/mr/reducer_count_final

