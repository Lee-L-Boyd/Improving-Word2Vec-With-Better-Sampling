import pickle
import tensorflow as tf
import os

test_list = [[1,2,3],[2,3,6],[1,3,10],[3,1,10],[3,2,6],[4,3,1], [3,4,1], [5,4,15], [4,5,15]]
batch_size = 4
num_epochs = 1000
def _int64_feature(val):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=val))

def _float64_feature(val):
  return tf.train.Feature(float_list=tf.train.FloatList(value=val))


def convert_to(batch):
  example = tf.train.Example(
    features = tf.train.Features(
      feature = {
        'source':_int64_feature(batch[0]),
        'context':_int64_feature(batch[1]),
        'count':_float64_feature(batch[2])
      }))
  return example

def convert():
  #p=pickle.load(open('coor_matrix.pickle','rb'))
  #print(len(test_list))
  p=test_list
  #p = list(zip(*p))
  #print(p)
  total = len(p)
  global num_batches
  if total%batch_size == 0:
    num_batches = total//batch_size
  else:
    num_batches = total// batch_size + 1
  batches = []
  print(num_batches)
  with tf.python_io.TFRecordWriter('dataset_test.tfrecords') as writer:
    for batchno in range(num_batches):
      start = batchno*batch_size
      end = min(total, start + batch_size)
      example = convert_to([list(a) for a in zip(*p[start:end])])
      writer.write(example.SerializeToString())
      if batchno%100 == 0:
        print(batchno * batch_size)
  return

def decode(dataset):
  features = tf.parse_single_example(
    dataset,
    features={
      'source':tf.VarLenFeature(tf.int64),
      'context':tf.VarLenFeature(tf.int64),
      'count':tf.VarLenFeature(tf.float32)
    })
  source = tf.cast(features['source'], tf.int32)
  context = tf.cast(features['context'], tf.int32)
  count = tf.cast(features['count'], tf.float32)
  return source, context, count

def inputs():
  with tf.name_scope('input'):
    dataset = tf.data.TFRecordDataset('dataset_test.tfrecords')
    dataset = dataset.map(decode)
    #dataset = dataset.map(augment)
    dataset = dataset.shuffle(4)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(1)
    iterator = dataset.make_one_shot_iterator()
  return iterator.get_next()
convert()


g = tf.Graph()

vocab_size = 20
embedding_size = 2
count_max = 10
scaling_factor = (3/4)
learning_rate = .05
tf.random.set_random_seed(1234)
with g.as_default():
  source, context, count = inputs()
  source = tf.squeeze(tf.sparse_tensor_to_dense(source))
  context = tf.squeeze(tf.sparse_tensor_to_dense(context))
  count = tf.squeeze(tf.sparse_tensor_to_dense(count))
  #batch = tf.transpose(tf.stack([tf.squeeze(source), tf.squeeze(context), tf.squeeze(count)]))
  source_embeddings = tf.Variable(
    tf.random_uniform([vocab_size, embedding_size], 1.0, -1.0),
    name="source_embeddings")
  context_embeddings = tf.Variable(
    tf.random_uniform([vocab_size, embedding_size], 1.0, -1.0),
    name="context_embeddings")
  source_biases = tf.Variable(tf.random_uniform([vocab_size], 1.0, -1.0),
    name='source_biases')
  context_biases = tf.Variable(tf.random_uniform([vocab_size], 1.0, -1.0),
    name="context_biases")
  source_embedding = tf.squeeze(tf.nn.embedding_lookup([source_embeddings], source))
  context_embedding = tf.squeeze(tf.nn.embedding_lookup([context_embeddings], context))
  source_bias = tf.nn.embedding_lookup([source_biases], source)
  context_bias = tf.nn.embedding_lookup([context_biases], context)
  weighting_factor = tf.minimum(1.0,tf.pow(
    tf.div(count, count_max),
    scaling_factor))
  embedding_product = tf.multiply(source_embedding, context_embedding)
  eps = tf.shape(embedding_product)[0]
  #eps2 = tf.shape(embedding_product)[1]
  embedding_sum = tf.cond(tf.size(embedding_product)>embedding_size, lambda:tf.reduce_sum(embedding_product,1), lambda:tf.squeeze(tf.reduce_sum(tf.expand_dims(embedding_product,0),1)))
  #tf.reshape(embedding_product, [eps, eps2+1])
  #eps = tf.expand_dims(embedding_product,0)
  #tf.reshape(embedding_product,[-1,1])
  #embedding_sum = tf.reduce_sum(embedding_product,0)
  log_cos = tf.log(count)
  distance_expr = tf.square(tf.add_n([embedding_sum, source_bias, context_bias, tf.negative(log_cos)]))
  #distance_expr = tf.square(tf.add_n([bias_sum, tf.negative(log_cos)]))
  single_losses = tf.multiply(weighting_factor, distance_expr)
  total_loss = tf.reduce_sum(single_losses)
  tf.summary.scalar("GloVe_loss", total_loss)
  optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(
    total_loss)
  summary = tf.summary.merge_all()
  combined_embeddings = tf.add(source_embeddings, context_embeddings,
    name="combined_embeddings")



  test = tf.reduce_sum(source)
  init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
summary_batch_interval = 2
total_steps = 0
with tf.Session(graph=g) as sess:
  sess.run(init_op)
  summary_writer = tf.summary.FileWriter("log_dir2", graph = sess.graph)
  for epoch in range(num_epochs*num_batches):
    summary_str, ces, o, tl, wf, de, sb, ep, es, lc, sb, cb, se, ce, epss=sess.run([summary, combined_embeddings, optimizer, total_loss, weighting_factor, distance_expr, source_bias, embedding_product, embedding_sum, log_cos, source_bias, context_bias, source_embedding, context_embedding, eps])
    print("EP " + str(ep))
    print("ES " + str(es))
    print("SE " + str(se))
    print("CE " + str(ce))
    print("EPS " + str(epss))
    print("SB " + str(sb))
    print("DE " + str(de))
    print("WF" + str(wf))
    print("TL" + str(tl))
    print("CES" + str(ces))
    if (total_steps + 1) % summary_batch_interval == 0:
      #summary_str = sess.run(summary, feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, total_steps)
      #current_embeddings = combined_embeddings.eval()
      #output_path = os.path.join(log_dir, "epoch{:03d}.png".format(epoch + 1))
      #generate_tsne(output_path, embeddings=current_embeddings)
    total_steps += 1
  summary_writer.close()
'''def generate_tsne(self, path=None, size=(10, 10), word_count=20, embeddings=None):
  #if embeddings is None:
  #  embeddings = self.embeddings
  from sklearn.manifold import TSNE
  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  low_dim_embs = tsne.fit_transform(embeddings[:word_count, :])
  #labels = self.words[:word_count]
  labels = [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20]
  return _plot_with_labels(low_dim_embs, labels, path, size)
'''


#sl, d_expr, cnt, wf, lc, b_sum, s_bias, c_bias, emb_sum, source_emb, context_emb, embedding_prod, source, test=sess.run([single_losses, distance_expr, count, weighting_factor, log_cos, bias_su$
#print("THIS IS TEST: " + str(b_sum) + '\n' + str(lc) + '\n' + str(wf) + '\n'+ str(d_expr) + '\n' + str(sl))

