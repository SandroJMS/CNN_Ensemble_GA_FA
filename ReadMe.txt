For Car Dataset:
    ConvolutionalNewralNetworksTraining.py:
        line 128: pred = Dense(4, activation='softmax')(out)
    ImageDemo.py: 
        line 48: classes = ['bus', 'car', 'truck','motorcycle']

For Weather Dataset:
    ConvolutionalNewralNetworksTraining.py:
        line 128: pred = Dense(11, activation='softmax')(out)
    ImageDemo.py: 
        line 48: classes = ['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']


How to use:
1 - Use SeparateDatasetFiles.py to separate the dataset folder, change the line 12 to the name of the dataset folder that contain all the classes folders.
2 - Use ConvolutionalNewralNetworksTraining.py to train and save all the five models, change the line 128 to the total number of classes.
3 - Use EnsembleWeightOptimization.py to generate the ensemble weights using Genetic Algorithm and Firefly Algorithm.
4 - Use ImageDemo.py to plot the weights and test the algorithms.

Remember: The objective of the ensemble functions are to optimize the Binary Cross Entropy on the validation data, this behaviour may cause overfitting, making the algorithm worse on other images.