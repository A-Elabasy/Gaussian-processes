# Gaussian-processes-using-libraries
Classification of MNIST dataset using libraries 
First if you do not have the data set the downlaod flag to true
The 'our_script.py' code provides the basic gaussian classifier.
In the 'resiz.py', each image becomes 8*8 instead of 28*28 which helps in improving the running time with little degrade of the performance.
In the 'PCA.py' each image becomes 8*1 which helps in improving the run time significantly, using suitable length-scale for the RBF kernel, the total performance will remain acceptable.
The 'Batch.py' provides implementation in pytorch. Batching the data, helps significantly in degrading the run time.
