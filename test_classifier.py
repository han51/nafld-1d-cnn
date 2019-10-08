import os
import numpy as np
import tensorflow as tf
from datagenerator import InputDataGenerator   
import scipy.io

os.chdir('/home/han51/data/DL/ucsdliver/Radiology_MS/code') # replace with your own directory
rf_dir = '../data/rf_without_tgc/'
result_dir = '../results/classifier/'
test_file_list = '../data/file_lists_with_labels_classifier/test.txt'
checkpoint_path = '../models/classifier/'
result_filename = '../results/classifier_without_tgc.mat'

input_dim1 = 1024   # first dimension of the input RF signal
input_dim2 = 1      # second dimension of the input RF signal
batch_size = 256
num_classes = 2     # = 1 for fat fraction estimator; = 2 for classifier
  
with open(test_file_list) as file_list:   
  lines = file_list.readlines()
  test_images = []
  test_patients = set()
  for l in lines:
    items = l.split()
    test_images.append(items[0][0:-10])
    test_patients.add(items[0][0:4])
test_patients = list(test_patients)
test_patients.sort()
test_patient_image_idx = list()
for pt in test_patients:
    test_patient_image_idx.append([i for i, s in enumerate(test_images) if pt in s])
                                     
test_generator = InputDataGenerator(test_file_list, rf_dir, rf_size=(input_dim1, batch_size), 
                                       shuffle = 0, num_classes = num_classes) 
test_batches_per_epoch = np.floor(test_generator.data_size*test_generator.rf_size[1]/batch_size).astype(np.int16)
test_generator.reset_pointer()
                                  
# Start Tensorflow session
sess = tf.Session()
saved_model = os.path.join(checkpoint_path, 'model_epoch50.ckpt.meta')
saved_checkpoint = os.path.join(checkpoint_path, 'model_epoch50.ckpt')
imported_graph = tf.train.import_meta_graph(saved_model)
imported_graph.restore(sess, saved_checkpoint)
                 
y_label_images = np.zeros([test_batches_per_epoch])
y_raw_matrix = np.zeros([test_batches_per_epoch, batch_size]) 
      
scores = tf.get_collection('prob')[0]
x = tf.get_collection('x')[0]

for imgi in range(test_batches_per_epoch):
    batch_test_x, batch_test_y = test_generator.next_batch(batch_size)
    y_label_images[imgi] = batch_test_y[:,1].mean()            
    y_raw_matrix[imgi,:] = sess.run(scores, feed_dict={x: batch_test_x}).T[1]

y_raw_images = y_raw_matrix.mean(1)
num_pt = test_patients.__len__()
y_label_pt = np.zeros(num_pt)
y_raw_pt = np.zeros(num_pt)
pti = 0;
for idx in test_patient_image_idx:
    y_label_pt[pti] = y_label_images[idx].mean()
    y_raw_pt[pti] = y_raw_images[idx].mean()
    pti = pti+1

# save the fat fraction estimator results in .mat format for statistical analysis in matlab
scipy.io.savemat(result_filename, dict(y_label_pt = y_label_pt, 
                                       y_label_images = y_label_images,
                                       y_raw_matrix = y_raw_matrix,
                                       y_raw_images = y_raw_images,
                                       y_raw_pt = y_raw_pt,
                                       pt_img_idx = test_patient_image_idx))        
sess.close()   
