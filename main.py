import sys, os, time
import numpy as np
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'')))
from rnn import rnn
from sequence_generator import generator
from evaluation import evaluation



def ParseInputFile(input):
    if input is None:
        print "The input file in none"
        return
    f = open(input, 'r')
    # skipping the first line (title)
    f.next()
    # dictionary of stocks: key - date (YYYY-MM-DD); value = tuple of (index, close rate). easy way to get date's value
    stocks_dict = dict()
    # array of stocks - each element is a tuple of (date, close rate). easy way to iterate continuous dates
    data = list(f)
    row_count = len(data)
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

# def build_X_train(size):
#     mat = np.zeros(shape=(size,size))
#     for i in range(size):
#         mat[i,i] = 1
#     return mat

def build_train_data(size, close_arr):
    x_train = np.zeros(shape=(len(close_arr), size), dtype=int)
    for i in range(len(close_arr)):
        x_train[i, close_arr[i]] = 1
    y_train = np.array(x_train[1:])
    return x_train, y_train


# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

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
    train_with_sgd(alg, X_train[:100], Y_train[:100], nepoch=1, evaluate_loss_after=1)

    learning_t = time.time()
    print 'learning time = ', learning_t-start_t

    generated_seq_num = 10

    # Data Generation
    gen = generator(alg, stocks_arr)
    generated_sequences = []
    for i in range(generated_seq_num):
        sent = []
        # We want long sentences, not sentences with one or two words
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
