import numpy as np
from math import log
from numpy import array
from numpy import argmax
import scipy.misc

def saveImage(filename,data):
   scipy.misc.imsave(filename+'.bmp', data.reshape((data.shape[0],data.shape[1])))
def checkcapacity(weights, y_, cap):
    y_ = np.argmax(y_, axis=-1)
    sum = 0.0
    for i in range(len(weights)):
        sum += weights[i] * y_[i]
    if sum <= cap:

        return True
    else:
        return False
def getcost(values, y_):
    sum = 0.0
    y_ = np.argmax(y_, axis=-1)
    for i in range(len(y_)):
        sum += values[i] * y_[i]
    return sum
def problems_accuracy(y, y_p, n,verbose=False):
    reward = 0
    y = np.argmax(y, axis=-1)
    y_p = np.argmax(y_p, axis=-1)
    for i in range(n):
        if y[i] == y_p[i]:
            if verbose:
                print (i,y[i], '-----', y_p[i])
            reward += 1
        else:
            if verbose:
                print ('WRONG' ,i,y[i], '-----' ,y_p[i])
    #print reward

    if reward == n:
        return 1, reward
    else:
        return 0, reward

def load_data(files):
    return np.load(files[0]) , np.load(files[1]) , np.load(files[2])
def beam_search_decoder(data, k):
	sequences = [[list(), 1.0]]
	# walk over each step in sequence
	for row in data:
		all_candidates = list()
		# expand each current candidate
		for i in range(len(sequences)):
			seq, score = sequences[i]
			for j in range(len(row)):
				candidate = [seq + [j], score * -log(row[j])]
				all_candidates.append(candidate)
		# order all candidates by score
		ordered = sorted(all_candidates, key=lambda tup:tup[1])
		# select k best
		sequences = ordered[:k]
	return sequences
