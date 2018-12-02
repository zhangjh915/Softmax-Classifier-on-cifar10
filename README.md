# Softmax-Classifier-with-Cross-Entropy-Loss
## Description:
A Softmax classifier with cross entropy loss and regularization on cifar10 images written in Python3.

This model uses cifar10 as the dataset.  The dataset can be downloaded following the instrustions in [The CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

## Model Performance:
The final accuracy of the model is 0.3606 and the learned weights are presented below.

![alt text](/images/weights.png)

## Softmax Classifier with Cross-Entropy Loss:
Softmax function takes an N-dimensional vector of real numbers and transforms it into a vector of real number in range (0,1). The softmax function outputs a probability distribution instead of one maximum value, making it suitable for probabilistic interpretation in classification tasks.

The softmax function is

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=p_{i}&space;=&space;\frac{e^{a_i}}{\sum_{k=1}^{N}e_k^a}" target="_blank"><img src="https://latex.codecogs.com/png.latex?p_{i}&space;=&space;\frac{e^{a_i}}{\sum_{k=1}^{N}e_k^a}" title="p_{i} = \frac{e^{a_i}}{\sum_{k=1}^{N}e_k^a}" /></a>
</p>


However, the exponential values can easily explode to an infinite large value in calculation for computers, e.g. e<sup>1000</sup>. In order to solve this problem, the max trick is used to restrict each exponential value within 1. For implementation, all of the exponential values are substracted by the maximum of them.

The cross entropy loss is defined as
<p align="center">
<img src="https://latex.codecogs.com/png.latex?L&space;=&space;-\frac{1}{N}&space;\sum_{i=1}^{N}log(p_i^{y_i})" title="L = -\frac{1}{N} \sum_{i=1}^{N}log(p_i^{y_i})" />
</p>

To calculate the analytical expression for the gradient of the loss function, the derivatives of the softmax function and cross entropy loss are needed. The details of the derivation are not presented in this post.

<p align="center">
<img src="https://latex.codecogs.com/png.latex?\frac{\partial&space;p_i}{\partial&space;a_j}=\left\{\begin{matrix}&space;p_i(1-p_i)&space;&&space;if&space;&&space;i=j\\&space;-p_j&space;p_i&space;&&space;if&space;&&space;i\neq&space;j&space;\end{matrix}\right." title="\frac{\partial p_i}{\partial a_j}=\left\{\begin{matrix} p_i(1-p_i) & if & i=j\\ -p_j p_i & if & i\neq j \end{matrix}\right." />
</p>

<p align="center">
<img src="https://latex.codecogs.com/png.latex?\frac{\partial&space;L}{\partial&space;o_i}=p_i-y\_oh_i" title="\frac{\partial L}{\partial o_i}=p_i-y\_oh_i" />
</p>

For the derivative of the cross entropy loss, *y_oh* is the one-hot representation of the labels.

Besides, the implementation also includes a regularization term *R(W)* multiplied with a regularization parameter <img src="https://latex.codecogs.com/png.latex?\lambda" title="\lambda" />. Mini-batch Stochastic Gradient Descent (SGD) is used to minimize the loss by computing the gradient w.r.t. a randomly selected batch from the training set. This method is more efficient than computing the gradient w.r.t the whole training set before each update is performed.

## Code Usage
1. Clone or download the code
2. Create a folder called "data" and unzip the downloaded cifar10 data in it
3. Run model_visualization.py

## Reference
1. [https://www.cc.gatech.edu/classes/AY2019/cs7643_fall/hw0-q8/](https://www.cc.gatech.edu/classes/AY2019/cs7643_fall/hw0-q8/).
2. [https://deepnotes.io/softmax-crossentropy](https://deepnotes.io/softmax-crossentropy).
