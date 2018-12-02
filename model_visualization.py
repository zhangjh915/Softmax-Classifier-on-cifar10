from load_data import load_dataset
from gradient_check import grad_check
from softmax_classifier import SoftmaxClassifier
import numpy as np
import matplotlib.pyplot as plt


def main():
    x_train, y_train, x_test, y_test = load_dataset()
    print('Training data shape: ', x_train.shape, '     Train labels shape: ', y_train.shape)
    print('Test data shape:     ', x_test.shape, '     Test labels shape: ', y_test.shape)
    print()

    classifier = SoftmaxClassifier()
    loss, grad = classifier.cross_entropy_loss(x_train, y_train, 1e-5)

    # Gradient check for the model
    f = lambda w: classifier.cross_entropy_loss(x_train, y_train, 0.0)[0]
    print('Gradient Check:')
    grad_check(f, classifier.W, grad, 10)
    print()

    # Plot the loss for the training
    loss_record = classifier.train(x_train, y_train, lr=1e-6, reg=1e4)
    plt.plot(loss_record)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()

    # Evaluation on test set
    y_test_pred = classifier.predict(x_test)
    accuracy = np.mean(y_test == y_test_pred)
    print('Accuracy of the Softmax classifier on the test set: %f' % accuracy)

    # Visualize the learned weights for each class
    w = classifier.W[:, :-1]  # Strip out the bias
    w = w.reshape(10, 32, 32, 3)

    w_min, w_max = np.min(w), np.max(w)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        plt.subplot(2, 5, i + 1)

        # Rescale the weights to be between 0 and 255 for image representation
        w_img = 255.0 * (w[i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(w_img.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])
    plt.show()


if __name__ == "__main__":
    main()
