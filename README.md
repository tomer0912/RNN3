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
- Then we took the column of the date and the column of the closing price and inserted the data into an array
- After that we reversed the array, Because we wanted the earliest time to be at the beginning And the most recent time to be at the end.
- Then we Rounded the price, We did so because we wanted a finite number of values And because The accuracy of the number wasn't as important as the the tendency.
- Because of how matrix multiplication works we can’t simply use a number as an input. Instead, we represent each number as a one-hot vector of size number of possible values which is the highest stock + delta.

###RNN Algorithm
the code for the RNN algorithm is attached to the GIT project.

###Generating sequences
the code for the generating sequences is attached to the GIT project.

###Building the Model and using it to generated sequences
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
Formally 

![](https://github.com/tomerse/RNN3/blob/master/MDE.PNG)

where x and y are sequence of length n.

###Results & Conclusions

The results of the research wasn't as we expected.


We expected the generated sequences to be similar to the original sequences, but the result was That the gaps between values in the generated sequences was larger than the gaps of the original sequences. 


We can conclude that maybe the model wasn't exactly fit for the data, Or maybe there was something missing in our implementation that caused it.
We looked for bugs but we couldn't find the problem.


The reason for the large Gaps in the generated sequences might come from the behavior of the model:
it gives equal Probability for each of the values, That means that the model doesn't Improves itself by learning - But we only did 10 iterations!
If we had more time all better hardware we Would have done Thousands of Iterations and then the model would improve itself by learning.
So the conclusion is that it happened because lack of Iterations, And not because of Problems with the model itself.







