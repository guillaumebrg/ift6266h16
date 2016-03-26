# ift6266h16

# Dogs vs Cats code :
- If you have Keras and Fuel installed on your computer, you should be able to launch a training using the folowing steps :
      - In training_params.py, make sure that data_access is set to "fuel"
      - Create a folder 'experiments' at the racine of 'train_model.py'
      - Open a console and run "python train_model.py -train"

- If you want to load datasets in memory, you should set "data_acces" to "in-memory" in training_params.py, and write the path to numpy files (saved using numpy.save) in "dataset_path" and "targets_path". The dataset must be an array of shape (25000, Size_X, Size_Y, Channels), and targets must be 1D (targets will be converted to 2D arrays during preprocessing). The code is going to divide the whole dataset into a trainset (17500 by default), validationset (3750), and a testset (3750). See dataset_division.py in dataset.py.

How does it work ?

- The train_model.py file contains the function 'launch_training' which is used for training a Keras model. This function instantiates a TrainingParams object (see training_params.py) which defines a lot of parameters for the training. 
- The architecture of the model should be defined in a python script located in the 'def_model' directory. This script should define a function called 'define_model' which create a keras network (see first_convnet.py for an example). This function should be then imported in the training_params.py file, and given to the TrainingParams class.
- The function 'train_model.py' is going to use a Keras callback to save models generated during the training. See the 'ModelChecpoint_perso' class.
- At the end of the training, a recap will be printed in a pdf file at the racine of the experiment directory, and a testset score is going to be stored in a textfile.

# Ipython notebooks description :

- :::::::::::::::: 1st assignment - MLP on MNIST database.ipynb ::::::::::::::::

Loading data, training curves, experiments on the batch size... Everything described in this blog post : 
https://guillaumebrg.wordpress.com/2016/01/21/first-assignment-mlp-implementation-applied-to-digits-recognition-mnist-database/

- :::::::::::::::: Theano Tutorial :::::::::::::::::::::

Tutorial take from here : http://deeplearning.net/tutorial/

- :::::::::::::::: Dog vs Cat - 1 - data visualization  :::::::::::::::::::::

Plot images from the dogs vs cats dataset - Compute some statistics on image sizes - Test of a preprocessing function operating data augmentation

- :::::::::::::::: Adversarial examples  :::::::::::::::::::::

Code for adversarial examples generation
