import numpy as np


class SoftmaxClassifier:
    def __init__(self):
        # Random initialization of W
        C, D = 10, 3073  # C :number of classes, D: dimension of each flattened image
        self.W = np.random.randn(C, D) * 0.001

    def train(self, x, y, lr=1e-5, reg=1e-3, num_iters=1000, batch_size=200):
        """
        Train the model using stochastic gradient descent.
        Arguments:
            x: D * N numpy array as the training data, where D is the dimension and N the training sample size
            y: 1D numpy array with length N as the labels for the training data
            lr: learning rate
            reg: regularization factor
            num_iters: number of iterations for training
            batch_size: batch size for the calculation of the mini-batch gradient descent
        Output:
            loss_record: Array containing the cross entropy loss history during the training process
        """
        _, N = x.shape  # N is the sample size

        # Train the model using mini-batch stochastic gradient descent.
        loss_record = []
        for it in range(num_iters):
            # Randomly chosen mini-batch
            indices = np.random.choice(N, batch_size, replace=True)
            x_batch = x[:, indices]
            y_batch = y[indices]

            # Calculate loss and gradient for the iteration
            loss, grad = self.cross_entropy_loss(x_batch, y_batch, reg)
            loss_record.append(loss)

            # Update W
            self.W -= lr * grad

        return loss_record

    def predict(self, x):
        """
        Predict labels using the trained model.
        Arguments:
            x: D * N numpy array as the test data, where D is the dimension and N the test sample size
        Output:
            y_pred: 1D numpy array with length N as the predicted labels for the test data
        """
        y = self.W.dot(x)
        y_pred = np.argmax(y, axis=0)
        return y_pred

    def cross_entropy_loss(self, x, y, reg):
        """
        Calculate the cross-entropy loss and the gradient for each iteration of training.
        Arguments:
            x: D * N numpy array as the training data, where D is the dimension and N the training sample size
            y: 1D numpy array with length N as the labels for the training data
        Output:
            loss: a float number of calculated cross-entropy loss
            dW: C * D numpy array as the calculated gradient for W, where C is the number of classes, and 10 for this model
        """

        # Calculation of loss
        z = np.dot(self.W, x)
        z -= np.max(z, axis=0)  # Max trick for the softmax, preventing infinite values
        p = np.exp(z) / np.sum(np.exp(z), axis=0)  # Softmax function
        L = -1 / len(y) * np.sum(np.log(p[y, range(len(y))]))  # Cross-entropy loss
        R = 0.5 * np.sum(np.multiply(self.W, self.W))  # Regularization term
        loss = L + R * reg  # Total loss

        # Calculation of dW
        p[y, range(len(y))] -= 1
        dW = 1 / len(y) * p.dot(x.T) + reg * self.W
        return loss, dW
