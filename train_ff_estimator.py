import os
import numpy as np
import tensorflow as tf
from datetime import datetime
import livernet_1d_cnn 
from datagenerator import InputDataGenerator   
 
os.chdir('/home/han51/data/DL/ucsdliver/Radiology_MS/code') # replace with your own directory
rf_dir = '../data/rf_without_tgc/'
training_file_list = '../data/file_lists_with_labels_ff_estimator/training.txt'
checkpoint_path = '../models/ff_estimator/'
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)
training_writer_path = checkpoint_path

input_dim1 = 1024   # first dimension of the input RF signal
input_dim2 = 1      # second dimension of the input RF signal

# Learning parameters
learning_rate = 0.005
wd = 0.000          # no weight decay was used in the manuscript
num_epochs = 50
batch_size = 256
num_classes = 1     # = 1 for fat fraction estimator; = 2 for classifier

# TF pla0eholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, input_dim1, input_dim2, 1], name='alines')
y = tf.placeholder(tf.float32, [None, num_classes], name='labels')
tf.add_to_collection('x', x)
tf.add_to_collection('y', y)

# Initialize model
conv1, conv2, conv3, fc1, fc2, parameters = \
  livernet_1d_cnn.inference(x, input_dim2, wd, num_classes, batch_size)

tf.add_to_collection('fc2', fc2)
tf.add_to_collection('conv1', conv1)
tf.add_to_collection('conv2', conv2)
tf.add_to_collection('conv3', conv3)
tf.add_to_collection('fc1', fc1)

# Op for calculating the loss
with tf.name_scope("metric"):
  MSE = tf.reduce_mean(tf.square(fc2-y), name = 'MSE')  
  tf.add_to_collection('losses', MSE)
  total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

# Train op
with tf.name_scope("train"):
  optimizer = tf.train.AdagradOptimizer(learning_rate)
  gradients = optimizer.compute_gradients(total_loss)
  train_op = optimizer.apply_gradients(gradients)
  
# Add gradients to summary  
for gradient, var in gradients:
  tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary  
for var in parameters:
  tf.summary.histogram(var.name, var)
  
# Add the loss to summary
tf.summary.scalar('losses', total_loss)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

train_generator = InputDataGenerator(training_file_list, rf_dir, rf_size=(input_dim1, batch_size),
                                     shuffle = 2, num_classes = num_classes)
train_batches_per_epoch = np.floor(train_generator.data_size*train_generator.rf_size[1]/batch_size).astype(np.int16)
train_generator.reset_pointer()
   
# Initialize the FileWriter
training_writer = tf.summary.FileWriter(training_writer_path)
    
# Initialize an saver for store model checkpoints
saver = tf.train.Saver()
    
# Start Tensorflow session
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for epoch in range(num_epochs):
    print("{}: Epoch number: {}".format(datetime.now(), epoch+1))
    step = 1
    while step <= train_batches_per_epoch:
        batch_train_x, batch_train_y = train_generator.next_batch(batch_size)
        sess.run(train_op, feed_dict={x: batch_train_x, y: batch_train_y})           
        step += 1
        
    train_generator.reset_pointer()       
        
  #save checkpoint of the model
  checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+'.ckpt')
  save_path = saver.save(sess, checkpoint_name)  
