# Code for manual hypter-parameter tuning within the training set via cross-validation 
# for the fat fraction estimator

# Use the following steps to compare N sets of hyper parameters
# 1) For each set of hyper parameters, manually specify the parameters in this file 
#    and the parameters and/or network architecture in livernet_hyper_para_tuning.py
# 2) Run this code to obtain mse_vector (i.e., mean square error each epoch)
# 3) Compare the values of mse_vector obtained from the N sets of hyper parameters.
#    This comparison will allow you to choose the best set of hyper parameters. 
#    The hyper parameter number of epochs is a built-in parameter for comparison.
 
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
import livernet_hyper_para_tuning
from datagenerator import InputDataGenerator  

os.chdir('/home/han51/data/DL/ucsdliver/Radiology_MS/code')   # replace with your own directory
rf_dir = '../data/rf_without_tgc/'                             
checkpoint_path = '../models_temp/ff_estimator/'             
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)
training_writer_path = checkpoint_path

input_dim1 = 1024   # first dimension of the input RF signal
input_dim2 = 1      # second dimension of the input RF signal

# Learning parameters
learning_rate = 0.005
wd = 0.000          # no weight decay was used in the manuscript
max_num_epochs = 100 
batch_size = 256
num_classes = 1    # = 1 for fat fraction estimator; = 2 for classifier
num_folds = 6      # number of folds used in cross-validation

# TF pla0eholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, input_dim1, input_dim2, 1], name='alines')
y = tf.placeholder(tf.float32, [None, num_classes], name='labels')
tf.add_to_collection('x', x)
tf.add_to_collection('y', y)

# Initialize model
conv1, conv2, conv3, fc1, fc2, parameters = \
  livernet_hyper_para_tuning.inference(x, input_dim2, wd, num_classes, batch_size)

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

mse_matrix = np.zeros([max_num_epochs, num_folds])  

for fold in range(0, num_folds):
  training_file_list = '../data/file_lists_with_labels_ff_estimator/training_t' + str(fold+1) + '.txt'
  val_file_list = '../data/file_lists_with_labels_ff_estimator/training_v' + str(fold+1) + '.txt'  
  
  train_generator = InputDataGenerator(training_file_list, rf_dir, rf_size=(input_dim1, batch_size),
                                     shuffle = 2, num_classes = num_classes)
    
  train_batches_per_epoch = np.floor(train_generator.data_size*train_generator.rf_size[1]/batch_size).astype(np.int16)
  train_generator.reset_pointer()

  val_generator = InputDataGenerator(val_file_list, rf_dir, rf_size=(input_dim1, batch_size),
                                     shuffle = 0, num_classes = num_classes)
    
  val_batches_per_epoch = np.floor(val_generator.data_size*val_generator.rf_size[1]/batch_size).astype(np.int16)
  val_generator.reset_pointer()
      
  with open(val_file_list) as file_list:   
    lines = file_list.readlines()
    val_images = []
    val_patients = set()
    for l in lines:
      items = l.split()
      val_images.append(items[0][0:-10])
      val_patients.add(items[0][0:4])
  val_patients = list(val_patients)
  val_patients.sort()
  val_patient_image_idx = list()
  for pt in val_patients:
    val_patient_image_idx.append([i for i, s in enumerate(val_images) if pt in s])
                                     
  # Start Tensorflow session
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(max_num_epochs):
      # training
      print("{}: Epoch number: {}".format(datetime.now(), epoch+1))
      step = 1
      while step <= train_batches_per_epoch:
        batch_train_x, batch_train_y = train_generator.next_batch(batch_size)
        sess.run(train_op, feed_dict={x: batch_train_x, y: batch_train_y})           
        step += 1    
      train_generator.reset_pointer()       

      # validation
      y_label_images = np.zeros([val_batches_per_epoch])
      y_raw_matrix = np.zeros([val_batches_per_epoch, batch_size]) 

      for imgi in range(val_batches_per_epoch):
        batch_val_x, batch_val_y = val_generator.next_batch(batch_size)
        y_label_images[imgi] = batch_val_y.mean()            
        y_raw_matrix[imgi,:] = sess.run(fc2, feed_dict={x: batch_val_x}).T
      val_generator.reset_pointer()
    
      y_raw_images = y_raw_matrix.mean(1)
      num_pt = val_patients.__len__()
      y_label_pt = np.zeros(num_pt)
      y_raw_pt = np.zeros(num_pt)
      pti = 0;
      for idx in val_patient_image_idx:
        y_label_pt[pti] = y_label_images[idx].mean()
        y_raw_pt[pti] = y_raw_images[idx].mean()
        pti = pti+1
      mse_matrix[epoch, fold] = ((y_raw_pt-y_label_pt)**2).mean()
    
    mse_vector = mse_matrix.mean(axis=1)
