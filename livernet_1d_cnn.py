# Final 1D-CNN architecture
import tensorflow as tf
import numpy as np
 
def _variable_biases(name, shape, value):
  var = tf.get_variable(name, shape, 
                        initializer=tf.constant_initializer(value, dtype=tf.float32), 
                        dtype=tf.float32)
  return var
 
# No weight decay was used for the Radiology publication. 
def _variable_with_weight_decay(name, shape, stddev, wd):
  var = tf.get_variable(name, shape, 
                        initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32),
                        dtype=tf.float32)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def inference(input_data, lateral_dim, WD, num_classes, batch_size):
  
  kernel_size1 = 16
  kernel_size2 = 8
  kernel_size3 = 8
  
  num_filter1 = 8
  num_filter2 = 8
  num_filter3 = 16
  num_filter5 = 32
  max_pool_size = 4 # both size and stride  
  bias_init = 0.1

  W_std1 = np.sqrt(2.0/kernel_size1/lateral_dim)/5.0
  W_std2 = np.sqrt(2.0/kernel_size2/num_filter1)
  W_std3 = np.sqrt(2.0/kernel_size3/num_filter2)
  W_std6 = np.sqrt(2.0/num_filter5)
  
  parameters = []
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay(name='weights', shape=[kernel_size1, lateral_dim, 1, num_filter1],
                                         stddev=W_std1, wd=WD)  
    conv = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding='SAME', data_format='NHWC')                         
    biases = _variable_biases('biases', [num_filter1], value=bias_init)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.tanh(pre_activation, name=scope.name)
    parameters += [kernel, biases]    
    
  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, max_pool_size, 1, 1], strides=[1, max_pool_size, 1, 1],
                         padding='SAME', name='pool1')
                         
  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay(name='weights',
                                         shape=[kernel_size2, 1, num_filter1, num_filter2],
                                         stddev=W_std2,
                                         wd=WD)
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME', data_format='NHWC', dilations=[1,1,1,1])
    biases = _variable_biases('biases', [num_filter2], value=bias_init) 
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.tanh(pre_activation, name=scope.name)
    parameters += [kernel, biases]

  # pool2
  pool2 = tf.nn.max_pool(conv2, ksize=[1, max_pool_size, 1, 1],
                         strides=[1, max_pool_size, 1, 1], padding='SAME', name='pool2')

  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay(name='weights', shape=[kernel_size3, 1, num_filter2, num_filter3],
                                         stddev=W_std3,
                                         wd=WD)
    conv = tf.nn.conv2d(pool2, kernel, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', dilations=[1,1,1,1])
    biases = _variable_biases('biases', [num_filter3], value=bias_init)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv3 = tf.tanh(pre_activation, name=scope.name)
    parameters += [kernel, biases]
  
  # pool3
  pool3 = tf.nn.max_pool(conv3, ksize=[1, max_pool_size, 1, 1],
                         strides=[1, max_pool_size, 1, 1], padding='SAME', name='pool3')
                         
  # fc1
  with tf.variable_scope('fc1') as scope: 
    flattened = tf.reshape(pool3, [batch_size, -1])  
    flat_len = flattened.get_shape()[1].value
    weights = _variable_with_weight_decay(name='weights', shape=[flat_len, num_filter5],
                                          stddev=np.sqrt(2.0/flat_len),
                                          wd=WD)
    biases = _variable_biases('biases', [num_filter5], value=bias_init)  
    fc1 = tf.tanh(tf.matmul(flattened, weights) + biases, name=scope.name)
    parameters += [weights, biases]    
      
  # fc2
  with tf.variable_scope('fc2') as scope: 
    weights = _variable_with_weight_decay(name='weights', shape=[num_filter5, num_classes],
                                          stddev=W_std6, wd=WD)  
    biases = _variable_biases('biases', [num_classes], value=bias_init)  
    fc2 = tf.nn.bias_add(tf.matmul(fc1, weights), biases, name=scope.name)   
    parameters += [weights, biases]    
    
  return conv1, conv2, conv3, fc1, fc2, parameters
