import tensorflow as tf
import numpy as np
import pickle
from tensorflow.contrib.tensorboard.plugins import projector
sess = tf.InteractiveSession()
#emb_file="part-whole_embs"
#emb2_file = "embs_custom.tgt"
r_dictionary = pickle.load(open('/home/student/Desktop/paper_1/hadoop-1.2.1/1_cache_files/200k_rdictionary' , 'r'))
LOG_DIR = 'log_dir/'
emb_file='/home/student/Desktop/paper_1/hadoop-1.2.1/1_sparse_matrix/pmi_nospan_reduced'
#emb2_file ='/home/student/Desktop/semantic_relations/embeddings/dataset_embeddings.emb'

embedding_size = 400


embedding_vector = []

embedding_vector = pickle.load(open(emb_file, "r"))

print(len(embedding_vector))
graph = tf.Graph()

#embedding_placeholder = tf.Variable(tf.float64, embedding_vector, trainable=False)
embedding_placeholder = tf.Variable(embedding_vector)
x = tf.placeholder(tf.float64, name = 'x')
tttt = tf.placeholder(tf.float64, name = 'tttt')
tf.summary.scalar('x', x)
tf.summary.scalar('tttt', tttt)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
init = tf.global_variables_initializer()


summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
init.run()
a = 3
b = 4
summary, post_x1 = sess.run([merged, tttt], feed_dict = {x: a, tttt: b})
train_writer.add_summary(tf.summary, 1)
train_writer.close()
saver = tf.train.Saver()


saver.save(sess, LOG_DIR+'model.ckpt', 1)
with open('metadata.tsv','w') as f:
    for i in range(len(embedding_vector)):
        f.write(r_dictionary[i]+'\n')
        #f.write('test' + str(i) + '\n')
    
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(summary_writer, config)
