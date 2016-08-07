import sys, os, time
import numpy as np
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'')))
from rnn import rnn
from sequence_generator import generator
from evaluation import evaluation
from training import training

# generated_seq_num - how many sequences to generate
generated_seq_num = 10

def ParseInputFile(input):
    if input is None:
        print "The input file in none"
        return
    f = open(input, 'r')
    # skipping the first line (title)
    f.next()
    # stocks_dict: dictionary of stocks: key -
    # date (YYYY-MM-DD); value = tuple of (index, close rate)
    # easy way to get date's value
    stocks_dict = dict()
    data = list(f)
    row_count = len(data)
    # stocks_arr: 2-D array of stocks with size row_count*row_count -
    # each [i,i] element is a combination of (date, close rate)
    # easy way to iterate continuous dates
    stocks_arr = np.empty(shape=(2,row_count),dtype=tuple)
    i = 0
    for line in data:
        rec = line.split(',')
        if len(rec) != 13:
            print "error in record: " + line
        else:
            date = datetime.strptime(rec[0], '%Y-%m-%d')
            close = int(np.round(float(rec[4])))
            value = (row_count-i-1, close)
            stocks_dict.update({date:value})
            stocks_arr[0, i] = date
            stocks_arr[1, i] = close
            i=i+1

    stocks_arr[0] = stocks_arr[0][::-1]
    stocks_arr[1] = stocks_arr[1][::-1]


    # test
    # index = stocks_dict.get(datetime.strptime('2016-08-04', '%Y-%m-%d'))[0]
    # print len(stocks_arr[1])
    # cl = stocks_arr[1][index] #   should be 105.87

    return stocks_arr, stocks_dict


def build_train_data(size, close_arr):
    x_train = np.zeros(shape=(len(close_arr), size), dtype=int)
    for i in range(len(close_arr)):
        x_train[i, close_arr[i]] = 1
    y_train = np.array(x_train[1:])
    return x_train, y_train







def main():

    if len(sys.argv) != 2:
        print "Error: incorrect number of args:" + str(len(sys.argv))
        return
    input = sys.argv[1]

    start_t = time.time()

    # Preprocessing
    stocks_arr, stocks_dict = ParseInputFile(input)
    np.random.seed(5)
    alg = rnn(stocks_arr, stocks_dict)
    X_train, Y_train = build_train_data(alg.interval, stocks_arr[1])

    # Learning RNN
    o, s = alg.fw_propagation(X_train[10])
    predictions = alg.predict(X_train[10])

    #alg.gradients_check([0, 1, 2, 3], [1, 2, 3, 4])

    tr = training(alg, X_train[:100], Y_train[:100])
    tr.train_with_sgd(iterations=1, evaluate_loss_after=1)

    learning_t = time.time()
    print 'learning time = ', learning_t-start_t


    # Data Generation
    gen = generator(alg, stocks_arr)
    generated_sequences = []
    for i in range(generated_seq_num):
        sent = []
        sent = gen.generate_sequence(size=10+i*10)
        generated_sequences.append(sent)
        print 'generated seq ' + str(i) + ': ' + str(sent)

    generation_t = time.time()
    print 'generation time = ', generation_t - learning_t

    # Evaluation
    eval = evaluation(stocks_arr)
    deltas = np.zeros(shape=generated_seq_num)
    for i in range(generated_seq_num):
        seq,delta = eval.evaluate_generated_sequence(generated_sequences[i])
        deltas[i] = delta
    avg = np.mean(deltas)
    print 'average delta = ' ,avg

    evaluation_t = time.time()
    print 'evaluation time = ', evaluation_t - generation_t

    print 'total time = ', evaluation_t - start_t



""" Invoking Main Function """
if __name__ == "__main__":
    main()
