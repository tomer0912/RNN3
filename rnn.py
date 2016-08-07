import numpy as np
import operator


# SGD - Stochastic Gradient Descent

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

class rnn(object):
    """RNN class - implementation for RNN algorithm"""

    def __init__(self, stocks_arr, stocks_dict, hidden_dim=100, bptt_trunc=10):
        self.stocks_arr = stocks_arr
        self.stocks_dict = stocks_dict
        max = (int)(np.max(self.stocks_arr, axis=1)[1]+5)
        # interval - range of valid values for the stock price
        self.interval = max
        # hidden_dim - the 'memory'. How deep the hidden layer remembers
        self.hidden_dim = hidden_dim
        # bptt - BACKPROPAGATION THROUGH TIME
        self.bptt_trunc = bptt_trunc
        # Randomly initialize the network parameters
        # U - handle inputs to hidden layer
        self.U = np.random.uniform(-np.sqrt(1. / self.interval), np.sqrt(1. / self.interval), (self.hidden_dim, self.interval))
        # V - handle hidden layer to output
        self.V = np.random.uniform(-np.sqrt(1. / self.hidden_dim), np.sqrt(1. / self.hidden_dim), (self.interval, self.hidden_dim))
        # W - handle between hidden layers
        self.W = np.random.uniform(-np.sqrt(1. / self.hidden_dim), np.sqrt(1. / self.hidden_dim), (hidden_dim, self.hidden_dim))


    def fw_propagation(self, x):
        #number of time steps
        steps = len(x)
        # s - array to save all hidden layers
        # add one additional element for the initial hidden, which we set to 0
        s = np.zeros((steps + 1, self.hidden_dim))
        #s[-1] = np.zeros(self.hidden_dim)
        # o - outputs at each time step
        o = np.zeros((steps, self.interval))
        for t in np.arange(steps):
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.fw_propagation(x)
        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        loss = 0
        for i in np.arange(len(y)):
            o, s = self.fw_propagation(x[i])
            correct_price_predictions = o[np.arange(len(y[i])), y[i]]
            loss += -1 * np.sum(np.log(correct_price_predictions))
        return loss

    def calculate_loss(self, x, y):
        # divide total loss by number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / N

    def back_propogation(self, x, y):
        size = len(y)
        o, s = self.fw_propagation(x)
        # accumulating gradients
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        for t in np.arange(size)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            dLdW, dLdU, delta_t = self.back_propogation_through_time(t, dLdW, dLdU, x, s, delta_t)
        return [dLdU, dLdV, dLdW]

    def back_propogation_through_time(self, t, dLdW, dLdU, x, s, delta_t):
        for bptt_step in np.arange(max(0, t - self.bptt_trunc), t + 1)[::-1]:
            # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
            dLdW += np.outer(delta_t, s[bptt_step - 1])
            dLdU[:, x[bptt_step]] += delta_t
            delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        return [dLdW, dLdU, delta_t]

    def sgd_step(self, x, y, learning_rate):
        # Calculate gradients
        dLdU, dLdV, dLdW = self.back_propogation(x, y)
        # update params according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW



    def gradients_check(self, x, y, h=0.001, err_th=0.01):
        # Calculate gradients using backpropagation in order to check if correct
        bptt_gradients = self.back_propogation(x, y)
        # parameters we want to check.
        params = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pid, pname in enumerate(params):
            self.gradient_check(x, y, h, err_th, bptt_gradients, pid, pname)


    def gradient_check(self, x, y, h, err_th, bptt_gradients, pid, pname):
        # Get the actual parameter value from the algorithm
        parameter = operator.attrgetter(pname)(self)
        print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            gradplus = self.calculate_total_loss([x], [y])
            parameter[ix] = original_value - h
            gradminus = self.calculate_total_loss([x], [y])
            estimated_gradient = (gradplus - gradminus) / (2 * h)
            # Reset parameter to original value
            parameter[ix] = original_value
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pid][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient) / (
                np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > err_th:
                print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                print "+h Loss: %f" % gradplus
                print "-h Loss: %f" % gradminus
                print "Estimated_gradient: %f" % estimated_gradient
                print "Backpropagation gradient: %f" % backprop_gradient
                print "Relative Error: %f" % relative_error
                return
            it.iternext()
        print "Gradient check for parameter %s passed." % (pname)




