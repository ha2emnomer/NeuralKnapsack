from Utils import checkcapacity
import numpy as np
def greedyKnapsack(values, weights, cap):
    ratios = [values[i] / float(weights[i]) for i in range(len(values))]
    indices = np.argsort(ratios)
    indices = np.flip(indices,0)
    ratios.sort()


    y = np.zeros((len(ratios), 2))
    for i in range(0, len(ratios)):
        # take item

        y[indices[i], 1] = 1
        y[indices[i], 0] = 0

        if checkcapacity(weights, y, cap):
            # if there still capacity take the item
            y[indices[i], 0] = 0
            y[indices[i], 1] = 1
        else:
            # don't take the item and stop 
            y[indices[i], 0] = 1
            y[indices[i], 1] = 0
            break
    return y
