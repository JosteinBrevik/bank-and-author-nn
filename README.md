# bank-and-author-nn
Homework for Machine Learning course at NTUST. Neural Network predictor for two datasets

This code is used to take in data from two csv datasets, preprocess this data and use it with sklearn's 
neural network to predict further results
Bank Management dataset: From UCI, contains info on people and their response to a marketing campaign
Spooky Author Identification dataset: Contains sentences from books by three different authors
Best results so far on a 80/20 training/testing split for both are 72,65% for BM and 99,97% for SAI

For SAI, an n-hot array approach was used. An "n-hot" array consists of all words used in all sentences, as long as that word appears a set amount of times (threshold). Each sentence is then turned into an array that displays how many times each word appears in that sentence.
Testing shows that a high threshold provides better results. On this dataset, a threshold of 20 meant that the resulting array contained 22124 words, whereas a threshold of 100 was only reached by 151. Best result was achieved with a threshold of 100 and a neural network with a single hidden layer at size 100. This predicted 99.97% (8377/8379) sentences correctly.
