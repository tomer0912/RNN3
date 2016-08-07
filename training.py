
import sys
from datetime import datetime


class training(object):
    """Training class - get an algorithm and training it"""

    # alg: The RNN model instance
    # X_train: The training data set
    # Y_train: The training data labels
    def __init__(self, alg, X_train, Y_train):
        self.alg = alg
        self.X_train = X_train
        self.Y_train = Y_train


    # learning_rate: Initial learning rate for SGD
    # iterations: Number of times to iterate through the complete dataset
    # evaluate_loss_after: Evaluate the loss after this many iterations
    def train_with_sgd(self, learning_rate=0.005, iterations=100, evaluate_loss_after=5):
        losses = []
        num_samples = 0
        for i in range(iterations):
            losses = self.calc_loss(i, evaluate_loss_after, losses, num_samples, learning_rate)

            for i in range(len(self.Y_train)):
                self.alg.sgd_step(self.X_train[i], self.Y_train[i], learning_rate)
                num_samples += 1


    def calc_loss(self, i, evaluate_loss_after, losses, num_samples, learning_rate):
        if (i % evaluate_loss_after == 0):
            loss = self.alg.calculate_loss(self.X_train, self.Y_train)
            losses.append((num_samples, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Loss after num_samples=%d iteration=%d: %f" % (time, num_samples, i, loss)
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        return losses