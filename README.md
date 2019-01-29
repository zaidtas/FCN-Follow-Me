# Project: Semantic Segmantation - Follow me
---

[//]: # (Image References)
[image0]: ./images/lr01.png
[image1]: ./images/lr001ep40.png
[image2]: ./images/lr001ep80.png
[image3]: ./images/following_target.png
[image4]: ./images/patrol_without_target.png
[image5]: ./images/patrol_with_target.png

## Problem

We have to train a FCN to do semantic segmentation on the images from a quadrotor in three classes background, other people and "our hero". This helps a quadrotor follow the hero around in the simulation.

## Network Architecture

We have learned in previous lessons that Fully Connected Networks (FCNs) have the ability to do classification on every pixel as opposed to simple CNNs that do classification on the whole image. They are made up of 3 different parts.  
* Encoder Network
* 1x1 convolution filter
* Decoder Network

We have equal number of decoder blocks as the number of encoder blocks so that the output layers are the same size at the original input image. FCNs also bear no constraint on the size of the input images.


### Encoder Network
It works as a normal CNN i.e. comprised of convolution layers that downsample the input, increasing depth but missing a fully connected network at the end and a subsequent softmax output layer. Instead the final layer is passed to a 1x1 convolutional layer.

The encoder network is comprised of encoder blocks which is basically a separable convolution layer in our network. Separable convolution layer is a technique for reducing the number of parameters in each convolution, thus increasing efficiency for the encoder network. They comprise of a "convolution performed over each channel of an input layer and followed by a 1x1 convolution that takes the output channels from the previous step and then combines them into an output layer." The encoder layers basically make a model learn features from a dataset by itself. Each layer learns a more complex feature compared to the previous layer. However, the deeper the network, the more computationally intensive it becomes to train.

### 1x1 Convolutional Layer

The 1x1 convolution layer is a regular convolution but with a kernel and stride of 1 essentially allowing the network to be able to retain spatial information from the encoder as opposed to fully connected layer.

### Decoder Network

It is comprised of decoder blocks. Each block is comprised of billinear upsampling layer with an upsampling factor of 2. Billinear upsampling takes the weighted average of the four nearest known pixels from the given pixel, estimating the new pixel intensity value. It is much more computationally efficient than transposed convolutional layers. The upsampled layer is concatenated with a layer with more spatial information than the upsampled one using skip connections. This is done to retain some of the finer details from the previous layers. We add separable convolution layers after this step for the model to be able to learn finer spatial details from the previous layers better.

### Model

We have chosen a model with 3 encoder layers, a 1x1 convolution, and then 3 decoder layers. Skip connections are employed as below:

	Features Layer (?, 160, 160, 3) - - |
	Encoder 1 (?, 80, 80, 32) - - - -|  |
	Encoder 2 (?, 40, 40, 64)  - -|  |  |
	Encoder 3 (?, 20, 20, 128)    |  |  |  Skip
	1x1 Conv (?, 20, 20, 128)     |  |  |  Connections
	Decoder 1 (?, 40, 40, 128) - -|  |  |
	Decoder 2 (?, 80, 80, 64) - - - -|  |
	Decoder 3 (?, 80, 80, 64) - - - - - |
	Output Layer (?, 160, 160, 3)

## Data Collection

We used the sample dataset provided to learn our models. We will then see how that leads to a poor performance as the sample training set doesn't contain all possible examples to learn.

## Hyperparameter selection
### Initial starting point

- **batch_size**: We used a batch size of 32 which seemed good for stochastic training. Can be lowered if the training seems unstable.
- **num_epochs**: Started with 40, usually changed if a particular model still learns beyond that.
- **steps_per_epoch**:  total number of images in training dataset divided by the batch_size. This remains constant throughout our analysis as this value makes sure we have gone through the entire dataset in one epoch.
- **validation_steps**: total number of images in validation dataset divided by the batch_size
- **workers**: 1  
- **learning_rate**: 0.01

We observe that for a learning rate of 0.01, the learning is not very stable as shown by an irratic loss on the validation data below:  ![LR01][image0]



We reduce the learning rate further to 0.001 and observe that altough the learning is very stable but the model is still learning after 40 epochs. The Score after 40 epochs is 0.3964 which is close to minimum required for passing evaluation. The learning curve is shown below: ![LR001EP40][image1]

We then double the epochs to 80 for sufficient learning and observe the following training curve and a score of 0.4313 which is above the minimum requirement. ![LR001EP80][image2]


The performances for a different range of learning rate and epochs is given below:

| LR  | epochs | Loss | Val Loss  | Score |
| ------------- |------| ------------- | ----- | ---- |
| 0.01| 40 | 0.0202 | 0.0306 | 0.3607 |
|0.001| 40 | 0.0173 | 0.0225 | 0.3964 |
|0.001| 50 | 0.0172 | 0.0314 | 0.4048 |
|0.001| 80 | 0.0152 | 0.0278 | 0.4313 |


## Results
We finally used the following hyperparameters

Epochs trained: 80
Batch Size: 32
Learning Rate: 0.001

Loss: 0.0152, Validation Loss: 0.0278

### IOU Scores
Left is image, Middle is Ground Truth, Right is Model prediction

*When Quad is following behind hero*
true positives: 539, false positives: 1, false negatives: 0

![follow][image3]

*When Quad is on patrol and no hero visable*
true positives: 0, false positives: 75, false negatives: 0

![follow][image4]

*When Quad can detect hero far away*
true positives: 154, false positives: 1, false negatives: 147

![follow][image5]


Weight: 0.7532
Final IoU: 0.5725
Final Score: 0.4313

## Discussion

The following could have been done to improve the final score:
- **Collect more diverse data** : We observe that our model does very well for following behind a hero but does pretty bad for the case of patrolling leading to a many false positives for the case of no hero visible and false negatives when the hero is far away. One way to improve score is to collect more data for the failing scenarios.

- **Have variable learning rate** : One thing that can be done is to have a decaying learning rate but the Adam optimizer provided to train our models already has an adaptive learning rate. The initial learning rate decides the upper limit.

- **Have more encoder and decoder layers** : The bigger the model, the better the learning but it also comes at a cost of being inefficient or overfitting the data.

The network can very well be trained to follow other objects like dog, cat, etc. we just need a ground truth labeled dataset to make the network learn this model for another class. In general labeled data collection is by itself a huge task FCNs for pixel wise classification.
