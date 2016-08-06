import numpy as np

# RMSE
def estimation_func(seq1, seq2):
    if len(seq1) != len(seq2):
        print 'error: different size of sequences'
        return
    n = len(seq1)
    sum = 0
    i = 1
    while i < n:
        cur_dif1 = seq1[i] - seq1[i-1]
        cur_dif2 = seq2[i] - seq2[i-1]
        sum += (cur_dif1 - cur_dif2)**2
        i += 1
    res = sum/n
    res = res**0.5
    return res

class evaluation(object):
    """Evaluation class - evaluate generated data with original data"""

    def  __init__(self, org):
        self.org = org[1]

    def evaluate_generated_sequence(self, dat):
        s = len(self.org)-len(dat)
        deltas = np.zeros(shape=s)
        for i in range(s):
            cur_seq = self.org[i:i+len(dat)]
            delta = estimation_func(cur_seq,dat)
            deltas[i] = delta
        start_index = np.argmin(deltas)
        return self.org[start_index:start_index+len(dat)], deltas[start_index]
