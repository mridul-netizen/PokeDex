# pokedex
![alt text](https://img.rankedboost.com/wp-content/uploads/2017/09/Pokemon-GO-GEN-4-Pokedex.png)
An image classification tool for predicting the type of pokemon provided to it
PROJECT POKEDEX
Machine learning and deep learning plays an important role in computer technology and artificial intelligence. With the use of deep learning and machine learning, human effort can be reduced in recognizing, learning, predictions and many more areas. This article presents recognizing the Pokémon’s from the dataset, convolution neural network on basis of performance, accuracy, time, and specificity.
 
Developing such a system includes a machine to understand and classify the images of Pokémon’s .Basically the dataset was from Kaggle which was later edited to meet the demands of project
So, before starting further deep in this topic, the better point should be to get familiar with the provided dataset. 
The two files of training and testing are:
1.	Training set images folder(PokeDex) 
1.	This folder further contains 150 folders of different pokemons
Containing their images
2.	Test set images files (test_data)
1.	It contains some random,non-arranged images of the pokemons from the training set.
## NN (Neural networks)

Neural Networks mimics the working of how our brain works. They have emerged a lot in the era of advancements in computational power.Deep learning is the acronym for Neural Networks, the network connected with multilayers. The layers are composited form nodes. A node is just a perception which takes an input performs some computation and then passed through a node’s activation function, to show that up to what context signal progress proceeds through the network to perform classification.
 
## CNN (Convolutional Neural Network)
Now let’s discuss the Convolutional Neural Networks, CNN has become famous among the recent times. CNN is part of deep, feed forward artificial neural networks that can perform a variety of task with even better time and accuracy than other classifiers, in different applications of image and video recognition, recommender system and natural language processing.
 
Use of CNN have spread as Facebook uses neural nets for their automatic tagging algorithms, google for photo search Amazon for their product recommendations, Pinterest for their home feed personalization and Instagram for search infrastructure. Image classification or object recognition is a problem is passing an image as a parameter and predicting whether a condition is satisfied or not (cat or not, dot or not), or the probability or most satisfying condition for an image. We are able to quickly recognize patterns, generalize from previous information and knowledge.
 
## Resnet-50
ResNet-50 is a convolutional neural network that is 50 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images. The network has an image input size of 224-by-224.

## Training Metrics
Metrics for training fastai models are simply functions that take input and target tensors, and return some metric of interest for training. You can write your own metrics by defining a function of that type, and passing it to Learner in the metrics parameter, or use one of the following pre-defined functions.

## Layers of Convolutional neural network
The multiple occurring of these layers shows how deep our network is, and this formation is known as the deep neural network.
1.	Input: raw pixel values are provided as input.
2.	Convolutional layer: Input layers translates the results of neuron layer. There is need to specify the filter to be used. Each filter can only be a 5*5 window that slider over input data and get pixels with maximum intensities.
3.	Rectified linear unit [ReLU] layer: provided activation function on the data taken as an image. In the case of back propagation, ReLU function is used which prevents the values of pixels form changing.
4.	Pooling layer: Performs a down-sampling operation in volume along the dimensions (width, height).
As we go deeper and deeper in the layers, the complexity is increased a lot. But it might worth going as accuracy may increase but unfortunately, time consumption also increases.

## PERFORMANCE MEASURES
In machine learning and deep learning, the performance or efficiency of a classifier is shown by various features which tells how well working the particular classifier is. As the names also suggest the measurements or values used to compare the performance of a classifier.

## Result
For research purpose, or applying the classifiers to real scenario problems. Accuracy and speed of recognition are considered the better measure. From the experiments, it is observed that the images are classified correctly even for the portion of the test images and the model was able to correctly predict the type of Pokémon provided at the particular index.
 
Overall comparison results
To show the accuracy, time, train loss and valid loss comparison among consecutive epochs used in the training set.

## Thank you

