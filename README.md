# Gaussian-processes.
Classification of MNIST dataset using libraries. 
First if you do not have the data, set the downlaod flag to true.
The 'our_script.py' code provides the basic gaussian classifier.
In the 'resiz.py', each image becomes 8x8 instead of 28x28 which helps in improving the running time with little degrade of the performance.
In the 'PCA.py' each image becomes 8x1 which helps in improving the run time significantly, using suitable length-scale for the RBF kernel, the total performance will remain acceptable.
The 'Batch.py' provides implementation in pytorch. Batching the data, helps significantly in degrading the run time.
The 'GPscratch.ipynb' provides gaussian process implementation on MNIST dataset as provided in gaussian process for machine learning reference.
The 'modelscomp.ipynb' provides comparison between the performance of different machine learning models on Mnist dataset.
'Script2.py' is an amplementtion of gaussian process on hippocampous dataset, the results provided by this model is .9 in training and .68 in testing.
