bin/hadoop jar contrib/streaming/hadoop-*streaming*.jar \
 -D stream.num.map.output.key.fields=3 \
 -D mapred.text.key.comparator.options='-k1 -k2 -k3' \
 -D mapred.text.key.partitioner.options='-k1' \
 -D map.output.key.field.separator='@#@' \
 -D mapred.reduce.tasks=0 \
 -D mapred.local.dir='\home\mr\local_dir' \
 -D mapred.work.output.dir='\home\mr\local_dir2' \
 -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
 -file ~/hadoop-1.2.1/reducer_echo_v2.py \
 -mapper ~/hadoop-1.2.1/reducer_echo_v2.py \
 -file ~/hadoop-1.2.1/reducer_unigram_count.py \
 -reducer ~/hadoop-1.2.1/reducer_unigram_count.py \
 -file ~/hadoop-1.2.1/reducer_unigram_count.py \
 -combiner ~/hadoop-1.2.1/reducer_unigram_count.py \
 -cacheFile /cache/top2.csv#top.csv \
 -input /test/tmp_tomatrix_data \
 -output /user/mr/test_tomatrix8

