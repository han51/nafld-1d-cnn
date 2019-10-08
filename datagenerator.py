# Prepare for the input data used in deep leanring models
# The original downsampled RF data were stored in .csv files,
#   each file containing a 1024 x 256 matrix (num_points per signal x num_signals),
#   and each patient having 10 csv files (=10 frames)

# class_list contains a list of csv file names and
#    the correpsonding labels (pdff values for ff_estimator and 0s and 1s for classifier)
 
import numpy as np

class InputDataGenerator:
    def __init__(self, class_list, rf_dir, shuffle = 0, rf_size=(4096, 256), 
                 num_classes = 2):
                
        # Init params
        self.n_classes = num_classes
        self.rf_dir = rf_dir
        self.shuffle = shuffle        
        self.rf_size = rf_size
        self.pointer = 0
        
        self.read_class_list(class_list)
        self.read_alines()       
        
        if self.shuffle !=0:
            self.shuffle_data()

    def read_class_list(self,class_list):

        with open(class_list) as f:
            lines = f.readlines()
            self.images = []
            self.image_labels = []
            for l in lines:
                items = l.split()
                self.images.append(items[0])
                if self.n_classes == 1:
                    self.image_labels.append(float(items[1]))
                else:
                    self.image_labels.append(int(items[1]))
            
            #store total number of data
            self.data_size = len(self.image_labels)
        
    def read_alines(self):
        self.rf = np.zeros([self.rf_size[0], self.rf_size[1]*self.data_size])  
        if self.n_classes == 1:
            self.labels = np.zeros([self.rf_size[1]*self.data_size], dtype=float) 
        else:
            self.labels = np.zeros([self.rf_size[1]*self.data_size], dtype=int)        
        col_beg = 0; i = 0;
        print("starts to load data")        
        for rfname in self.images:
            self.rf[:, col_beg:col_beg+self.rf_size[1]] = np.genfromtxt(self.rf_dir+ rfname, delimiter=',')            
            self.labels[col_beg:col_beg+self.rf_size[1]] = self.image_labels[i]            
            i += 1            
            col_beg += self.rf_size[1]
            #print(rfname)            
        print("finishes loading data")
        
    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        rf = self.rf.copy()
        labels = self.labels.copy()
 
        #create list of permutated index and shuffle data accoding to list
        if self.shuffle == 1: # shuffle by image
            tmp = np.arange(len(labels)).reshape(-1, self.rf_size[1])        
            idx = np.random.permutation(tmp).reshape(len(labels))
        else:  # shuffle by line
            idx = np.random.permutation(len(labels))
        self.rf = rf[:, idx]
        self.labels = labels[idx]
        
    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0
        
        if self.shuffle !=0:
            self.shuffle_data()
          
    def next_batch(self, batch_size):
        
        idx_beg = self.pointer
        idx_end = self.pointer + batch_size        
        self.pointer += batch_size
        
        alines = np.ndarray([batch_size, self.rf_size[0], 1, 1])
        for i in xrange(idx_beg,idx_end):
            alines[i-idx_beg] = self.rf[:, i].reshape([self.rf_size[0], 1, 1])                
            
        alines = alines.astype(np.float32)     
        labels = self.labels[idx_beg:idx_end]
        one_hot_labels = np.zeros((batch_size, self.n_classes))  
        if self.n_classes == 2:
            for i in range(len(labels)):
                one_hot_labels[i][labels[i]] = 1
        elif self.n_classes == 1:  # continous
            for i in range(len(labels)):
                one_hot_labels[i] = labels[i]          
        return alines, one_hot_labels
