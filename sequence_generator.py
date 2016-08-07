import numpy as np


class generator(object):
    """Generator class - generate data by rnn algorithm"""

    def __init__(self, alg, stocks_arr):
        self.alg = alg
        self.start_prediction_stock = stocks_arr[1][-1]


    def generate_sequence(self, size=10):
        # starting the predictions
        predictions = [self.start_prediction_stock]
        # Predict 'size' values
        for i in range(size):
            next_val_probs = self.alg.fw_propagation(predictions)[0]
            sampled_val = -1
            # predict until we get a valid value
            while sampled_val < 0 or sampled_val > self.alg.interval:
                samples = np.random.multinomial(1, next_val_probs[-1])
                sampled_val = np.argmax(samples)
            predictions.append(sampled_val)
        return predictions