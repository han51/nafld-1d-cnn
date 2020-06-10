# nafld-1d-cnn
1D-CNN models for NAFLD diagnosis and liver fat fraction quantification using radiofrequency (RF) ultrasound signals

1. The code is used for developing, training, and testing two 1D-CNN models: 
a) a classifier that differentiates between NAFLD and control (no liver disease); and
b) a fat fraction estimator that predicts the liver fat fraction. 
Both models use the radiofrequency ultrasound signals as the input and use the MRI-proton density fat fraction (PDFF) as the reference (labels). In the case of the classifier, NAFLD is defined as MRI-PDFF >= 5%.

2. livernet_1d_cnn.py contains the final model architecture for both the classifier and the fat fraction estimator. 

3. For model training and hyper parameter tuning, use hyper_parameter_tuning_classifier.py and hyper_parameter_tuning_ff_estimator.py

4. For final model training, use train_classifier.py and train_ff_estimator.py.

5. For model testing, use test_classifier.py and test_ff_estimator.py.

6. The tool datagenerator.py prepares for the input data used in deep learning models. The original downsampled RF data should be stored in .csv files, each file containing an RF frame represented by a 1024 x 256 matrix (num_points per RF signal x num_signals) and each patient having 10 csv files (=10 frames). This tool requires a file that contains a list of csv file names and the correpsonding labels (pdff values for the ff_estimator and 0s and 1s for the classifier).

7. The matlab script (stat_analysis.m) and R script (auc_plot_and_test.R) can be used for statistical analysis of the model performances.

# Contact
Aiguo Han (han51 at illinois.edu)

# Citation
If you use our code for publications, we would appreciate if you cite our paper:
A. Han, M. Byra, E. Heba, M. P. Andre, J. W. Erdman Jr, R. Loomba, C. B. Sirlin, and W. D. O’Brien Jr. "Noninvasive diagnosis of nonalcoholic fatty liver disease and quantification of liver fat with radiofrequency ultrasound data using one-dimensional convolutional neural networks." Radiology 295, no. 2 (2020): 342-350.

BibTeX format: 
@article{han2020noninvasive,
  title={Noninvasive diagnosis of nonalcoholic fatty liver disease and quantification of liver fat with radiofrequency ultrasound data using one-dimensional convolutional neural networks},
  author={Han, Aiguo and Byra, Michal and Heba, Elhamy and Andre, Michael P and Erdman Jr, John W and Loomba, Rohit and Sirlin, Claude B and O’Brien Jr, William D},
  journal={Radiology},
  volume={295},
  number={2},
  pages={342--350},
  year={2020},
  publisher={Radiological Society of North America}
}
