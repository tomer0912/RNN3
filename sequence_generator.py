import numpy as np


class generator(object):
    """Generator class - generate data by rnn algorithm"""

    def __init__(self, alg, stocks_arr):
        self.alg = alg
        self.start_prediction_stock = stocks_arr[1][-1]


    def generate_sequence(self, size=10):
        # We start the sentence with the start token
        predictions = [self.start_prediction_stock]
        #new_sentence = [word_to_index[sentence_start_token]]
        # Repeat until we get an end token
        for i in range(size):
            next_val_probs = self.alg.forward_propagation(predictions)[0]
            sampled_val = -1
            # We don't want to sample unknown words
            while sampled_val < 0 or sampled_val > self.alg.interval:
                samples = np.random.multinomial(1, next_val_probs[-1])
                sampled_val = np.argmax(samples)
            predictions.append(sampled_val)
        #sentence_str = [predictions[x] for x in predictions[1:-1]]
        return predictions