---
title: "Final project"   
author: "Tomer Belzer & Tomer Segal"   
date: "August 7, 2016"   
output: html_document
---
#Recurrent Neural Network
##the project's substance:
###Choosing a data set:
We choose to use an existing data set.
###Data Description:
The data set which ocean is Apple's stock price between the years 1981-2016.
The main challenge was to find this data set. But once we found it, it wasn't very difficult to work with it 
The research is interesting because ‫predicting the price of a stock successfully can make a lot of money for the people predicting it

###Pre-processing stages:
- First we had to read from a file.
```python
    f = open(input, 'r')
    # skipping the first line (title)
    f.next()
    data = list(f)
    row_count = len(data)
    # stocks_arr: 2-D array of stocks with size row_count*row_count -
    # each [i,i] element is a combination of (date, close rate)
    # easy way to iterate continuous dates
    stocks_arr = np.empty(shape=(2,row_count),dtype=tuple)
    i = 0
    for line in data:
        rec = line.split(',')
```
- Then we took the column of the date and the column of the closing price and inserted the data into an array
- Then we Rounded the price, We did so because we wanted a finite number of values And because The accuracy of the number wasn't as important as the the tendency.
```python
            date = datetime.strptime(rec[0], '%Y-%m-%d')
            close = int(np.round(float(rec[4])))
```
- After that we reversed the array, Because we wanted the earliest time to be at the beginning And the most recent time to be at the end.
```python
    stocks_arr[0] = stocks_arr[0][::-1]
    stocks_arr[1] = stocks_arr[1][::-1]
```
- Because of how matrix multiplication works we can’t simply use a number as an input. Instead, we represent each number as a one-hot vector of size number of possible values which is the highest stock + delta.
```python
def build_train_data(size, close_arr):
    x_train = np.zeros(shape=(len(close_arr), size), dtype=int)
    for i in range(len(close_arr)):
        x_train[i, close_arr[i]] = 1
    y_train = np.array(x_train[1:])
    return x_train, y_train
```

###RNN Algorithm
####Initialization
We start by declaring a RNN class an initializing our parameters. We initialize them randomly.
```python
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
```
####Forward Propagation

Predicting stock price probabilities

```python
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
```

####Calculating the Loss

To train our network we need a way to measure the errors it makes. Our goal is find the ideal parameters that minimize the loss function, we choose the following loss function 

![](https://github.com/tomer0912/RNN3/blob/master/loss-function.PNG)

```python
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
```



###Generating sequences

generating sequences

```python
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
```

###Building the Model and using it to generated sequences
building the model

```python
    # Preprocessing
    stocks_arr, stocks_dict = ParseInputFile(input)
    np.random.seed(5)
    alg = rnn(stocks_arr, stocks_dict)
    X_train, Y_train = build_train_data(alg.interval, stocks_arr[1])
    # Learning RNN
    o, s = alg.fw_propagation(X_train[10])
    predictions = alg.predict(X_train[10])
    tr = training(alg, X_train[:100], Y_train[:100])
    tr.train_with_sgd(iterations=1, evaluate_loss_after=1)
```
using the model to generate sequences 

```python
    # Data Generation
    gen = generator(alg, stocks_arr)
    generated_sequences = []
    for i in range(generated_seq_num):
        sent = []
        sent = gen.generate_sequence(size=10+i*10)
        generated_sequences.append(sent)
        print 'generated seq ' + str(i) + ': ' + str(sent)
        
```

evaluate the generated sequences

```python
    # Evaluation
    eval = evaluation(stocks_arr)
    deltas = np.zeros(shape=generated_seq_num)
    for i in range(generated_seq_num):
        seq,delta = eval.evaluate_generated_sequence(generated_sequences[i])
        deltas[i] = delta
    avg = np.mean(deltas)
    print 'average delta = ' ,avg
```

we built the model and ran it while using it for generating sequences of the stock price.
Those are the generated sequences

**generated seq 0:** [106, 333, 44, 329, 425, 367, 668, 44, 483, 644, 381]

**generated seq 1:** [106, 612, 475, 375, 507, 265, 398, 466, 445, 317, 370, 193, 405, 705, 105, 554, 83, 685, 296, 97, 250]

**generated seq 2:** [106, 662, 659, 496, 512, 31, 504, 599, 170, 547, 539, 408, 678, 54, 140, 609, 35, 41, 423, 589, 250, 395, 706, 497, 479, 105, 267, 261, 588, 587, 15]

**generated seq 3:** [106, 83, 216, 505, 525, 405, 21, 654, 94, 166, 687, 384, 657, 676, 261, 694, 258, 360, 487, 553, 291, 140, 523, 672, 402, 102, 380, 408, 587, 683, 420, 186, 133, 473, 277, 437, 343, 245, 548, 702, 9]

**generated seq 4:** [106, 121, 79, 336, 529, 397, 285, 364, 462, 81, 440, 334, 261, 499, 636, 704, 74, 261, 412, 605, 656, 445, 348, 81, 191, 133, 492, 392, 247, 68, 517, 114, 48, 52, 700, 23, 379, 445, 512, 456, 294, 336, 372, 567, 308, 494, 611, 101, 53, 230, 242]

**generated seq 5:** [106, 612, 591, 493, 218, 183, 637, 586, 650, 135, 101, 454, 583, 291, 405, 458, 669, 647, 373, 621, 74, 52, 501, 496, 408, 77, 540, 5, 607, 639, 10, 579, 560, 9, 370, 29, 614, 494, 462, 82, 677, 541, 360, 461, 632, 265, 23, 222, 416, 506, 189, 334, 601, 636, 375, 269, 58, 208, 603, 307, 453]

**generated seq 6:** [106, 405, 256, 490, 320, 30, 659, 55, 68, 334, 307, 552, 2, 656, 663, 509, 336, 36, 548, 242, 454, 215, 310, 191, 194, 121, 694, 67, 163, 208, 492, 416, 21, 435, 401, 634, 455, 512, 296, 260, 415, 105, 303, 239, 44, 614, 108, 659, 587, 226, 673, 621, 307, 133, 259, 89, 284, 660, 288, 226, 456, 45, 393, 356, 598, 221, 299, 417, 641, 117, 308]

**generated seq 7:** [106, 614, 361, 381, 208, 177, 188, 183, 601, 552, 467, 262, 5, 188, 664, 386, 132, 689, 647, 580, 524, 688, 655, 676, 643, 405, 21, 59, 674, 312, 368, 654, 179, 188, 641, 86, 685, 350, 584, 27, 319, 442, 557, 199, 288, 340, 555, 688, 596, 357, 426, 580, 672, 237, 492, 356, 314, 503, 660, 385, 640, 212, 308, 413, 144, 289, 648, 358, 573, 593, 444, 662, 291, 539, 600, 500, 473, 596, 244, 230, 383]

**generated seq 8:** [106, 442, 692, 301, 7, 160, 34, 2, 137, 538, 57, 163, 466, 556, 205, 134, 522, 247, 204, 705, 83, 507, 542, 305, 148, 220, 397, 18, 603, 512, 550, 224, 155, 205, 284, 224, 177, 401, 542, 587, 471, 193, 649, 576, 429, 191, 335, 596, 72, 415, 582, 129, 61, 16, 582, 447, 155, 273, 456, 525, 523, 163, 388, 175, 315, 44, 191, 330, 274, 80, 413, 46, 92, 231, 438, 556, 212, 367, 198, 267, 44, 234, 479, 287, 52, 356, 169, 418, 171, 513, 208]

**generated seq 9:** [106, 284, 94, 79, 84, 511, 359, 353, 490, 298, 248, 504, 46, 44, 536, 445, 156, 339, 436, 55, 674, 511, 240, 299, 513, 453, 483, 582, 591, 153, 55, 584, 625, 92, 548, 168, 632, 85, 213, 312, 18, 524, 182, 20, 161, 406, 375, 16, 430, 36, 89, 305, 41, 412, 12, 162, 649, 559, 642, 631, 703, 674, 197, 533, 217, 340, 128, 219, 372, 123, 637, 64, 188, 22, 401, 577, 505, 286, 261, 276, 336, 235, 230, 413, 163, 641, 286, 501, 676, 580, 84, 265, 128, 367, 86, 694, 496, 9, 549, 506, 583]


###Imagination measurement
For the task of evaluating the reclaimed data's quality we used the next Imagination measurement:
Two sequences are 'alike' if the tendency of both is the same. For example the next sequences are alike (1,2,3) (4,5,6).

Formally (MDSE = mean derivative squared error) 

![](https://github.com/tomer0912/RNN3/blob/master/MDE.PNG)

where x and y are sequence of length n.

the avarage of the minimum MDSE of each generated sequence with the original sequencess is 270.193902834

MDSE calculation:

```python
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
```

###Results & Conclusions

The results of the research wasn't as we expected.


We expected the generated sequences to be similar to the original sequences, but the result was That the gaps between values in the generated sequences was larger than the gaps of the original sequences. 


The reason for the large Gaps in the generated sequences might come from the behavior of the model:
it gives equal Probability for each of the values, That means that the model doesn't Improves itself by learning - But we only did 10 iterations!
If we had more time all better hardware we Would have done Thousands of Iterations and then the model would improve itself by learning.
So the conclusion is that it happened because lack of Iterations, And not because of Problems with the model itself.







