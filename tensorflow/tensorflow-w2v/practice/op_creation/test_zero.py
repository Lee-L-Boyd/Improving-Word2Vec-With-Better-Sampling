import tensorflow as tf
import os

zero_out_module = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)),'zero_out.so'))



#zero_out = _zero_out_module.zero_out
'''# Prints
array([[1, 0],
       [0, 0]], dtype=int32)'''

zero_out_module = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)),'zero_out.so'))
with tf.Session(''):
  result = zero_out_module.zero_out([5, 4, 3, 2, 1],3)
  #assertAllEqual(result.eval(), [5, 0, 0, 0, 0])
  print(result.eval())
