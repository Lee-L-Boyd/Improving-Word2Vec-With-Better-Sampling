../bin/hadoop fs -rmr /user/mr/word_count
../bin/hadoop jar ../contrib/streaming/hadoop-*streaming*.jar \
 -D stream.num.map.output.key.fields=2 \
 -D mapred.text.key.comparator.options='-k1' \
 -D mapred.text.key.partitioner.options='-k1' \
 -D mapred.reduce.tasks=1 \
 -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
 -file ~/hadoop-1.2.1/1_run_mr/m_word_count.py \
 -mapper ~/hadoop-1.2.1/1_run_mr/m_word_count.py \
 -file ~/hadoop-1.2.1/1_run_mr/r_word_count.py \
 -reducer ~/hadoop-1.2.1/1_run_mr/r_word_count.py \
 -file ~/hadoop-1.2.1/1_run_mr/r_word_count.py \
 -combiner ~/hadoop-1.2.1/1_run_mr/r_word_count.py \
 -input /user/mr/corpus/*/*/* \
 -output /user/mr/word_count
rm ~/hadoop-1.2.1/1_cache_files/word_count
../bin/hadoop fs -copyToLocal /user/mr/word_count/part-00000 \
 ~/hadoop-1.2.1/1_cache_files/word_count
python ../1_cache_files/get_top_200k.py
