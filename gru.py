from keras.constraints import maxnorm, Constraint
from keras.layers import Input, Masking, GRU, merge, Dense, TimeDistributed, BatchNormalization, Activation, Dropout,     Conv1D, RepeatVector ,Lambda
from keras.models import Model , Sequential
from keras.layers import concatenate,Bidirectional
from keras.layers.merge import Add, Dot , Multiply
from keras.initializers import RandomUniform, RandomNormal
import keras.backend as K
import numpy as np
from keras.optimizers import RMSprop, Adam, SGD,Adadelta
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping,CSVLogger,LearningRateScheduler, RemoteMonitor
from keras import regularizers
from time import time
import math
from imports.clr_callback import CyclicLR

# In[3]:

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
            # don't take the item
            y[indices[i], 0] = 1
            y[indices[i], 1] = 0
    return y


# In[4]:

def checkcapacity(weights, y_, cap):

    y_ = np.argmax(y_, axis=-1)
    sum = 0.0
    for i in range(len(weights)):
        sum += weights[i] * y_[i]

    if sum <= cap:

        return True
    else:
        return False



# In[5]:

def getcost(values, y_):
    sum = 0.0
    y_ = np.argmax(y_, axis=-1)
    for i in range(len(y_)):
        sum += values[i] * y_[i]
    return sum


# In[6]:

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


# In[7]:

def load_data(files):
    return np.load(files[0]) , np.load(files[1]) , np.load(files[2])
optimizer = Adam(lr=4e-3,clipnorm=1.0)
#optimizer  = RMSprop(clipnorm=1.0)
np.random.seed(50)
hidden = 256
size='large'
model = Sequential()
model.add(GRU(hidden,input_shape=(None,3), return_sequences=True))
model.add(GRU(hidden,kernel_initializer=RandomUniform(-0.08, 0.08), return_sequences=True))
model.add(GRU(hidden,kernel_initializer=RandomUniform(-0.08, 0.08), return_sequences=True))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])
approx_list =[500.0]
wait = 0
def evaluate(epoch,log):
    y_p = np.zeros((MAX_N,2))
    acc = 0
    gru_feas = 0
    feas_greedy = 0
    solver_costs = 0
    gru_costs = 0
    greedy_costs = 0
    total_correct_variables = 0
    infeas = 0
    #max_v = np.amax(x_w_c)
    #print max_v
    test_set = x_val.shape[0]
    for _i in range(test_set):
        #_i = probs[__i]
        weights = []
        values = []
        for t in range(MAX_N):
            weights.append(x_val[_i, t, 1]*10.0)
            values.append(x_val[_i, t, 0]*10.0)
        #x_val_t = np.zeros((1,N,1))
        #x_w_c_t = np.zeros((1,N+2,1))
        #x_val_t[:,:,0] = x_val[_i,:,0] / max_1
        #x_val_t = x_val / 19600.0
        #x_w_c_t[:,:,0] = x_w_c[_i,:,0] / max_2
        #x_w_c_t[:,-2,0] = -1.0
        y_p = model.predict(x_val[_i:_i+1])[0]
        #print sum(weights) ,  x_val[_i, 0, 2]
        if checkcapacity(weights, y_p, x_val[_i, 0, 2]*10.0):
            gru_costs+= getcost(values,y_p)
            gru_feas+=1
            y = greedyKnapsack(values, weights, x_val[_i, 0, 2]*10.0)
            if checkcapacity(weights, y, x_val[_i, 0, 2]*10.0):
                feas_greedy += 1
                greedy_costs += getcost(values, y)
                #optimial
                solver_costs += obj_val[_i]
                es , reward = problems_accuracy(y_val[_i], y_p, MAX_N)
                if es == 1:
                    acc += 1
                total_correct_variables += reward
        else:
            #continue

            infeas+=1
            indices = np.argsort(weights)
            indices = np.flip(indices,0)
            for w,v in enumerate(indices):
                #print _i, w
                if np.argmax(y_p[w]) == 1:
                    y_p[w,0] = 1
                    y_p[w,1] = 0
                    if checkcapacity(weights, y_p,x_val[_i, 0, 2]*10.0):
                        gru_costs+= getcost(values,y_p)
                        gru_feas+=1
                        y = greedyKnapsack(values, weights,x_val[_i, 0, 2]*10.0)
                        feas_greedy += 1
                        greedy_costs += getcost(values, y)
                        solver_costs += obj_val[_i]
                        es , reward = problems_accuracy(y_val[_i], y_p, MAX_N)
                        if es == 1:
                            acc += 1
                        total_correct_variables += reward
                        break

    gru_approx = max(solver_costs / gru_costs, gru_costs / solver_costs)
    #if len(approx_list) == 0:
    #    approx_list = [500.0]
    global wait
    global  approx_list
    if gru_approx >= approx_list[-1]:

        if wait >= 10:
            model.stop_training = True
            wait = 0

            #approx_list = [500.0]
            #approx_list.clear()

        else:
            wait+=1
            approx_list.append(gru_approx)

    else:
        model.save('.models/gru-'+ str(MAX_N)+size+'.hdf5')
        approx_list.append(gru_approx)
    print '\r'
    print infeas
    print('total number of correct variables:' , total_correct_variables / float(test_set*MAX_N))
    print ('number of feasible sols (greedy):', feas_greedy/ float(test_set))
    print ('greedy costs:', greedy_costs)
    print ('solver costs:', solver_costs)
    print ('GRU costs:', gru_costs)
    print ('accuracy:', acc / float(test_set))
    print ('number of feasible sols(GRU):', gru_feas / float(test_set))
    print ('Approx. ratio for GRU:', max(solver_costs / gru_costs, gru_costs / solver_costs))
    print ('Approx. ratio for greedy algorithm:', max(solver_costs / greedy_costs, greedy_costs / solver_costs))
model.summary()
MAX_N =5
x_train_v_w_c_f ,obj_train, y_train_f = load_data(['data/train/vwc_train_5.npy', 'data/train/objectives_train_5.npy','data/train/output_train_5.npy'])
x_val , obj_val , y_val = load_data(['data/test/vwc_test_5.npy', 'data/test/objectives_test_5.npy', 'data/test/output_test_5.npy'])
evaluate_callback = LambdaCallback(on_epoch_end=evaluate)
x_train_v_w_c_f /= 10.0
x_val /= 10.0
csv_logger = CSVLogger('logs/training_gru'+str(MAX_N)+'-'+size+'VARs_units.log')
model.fit(x_train_v_w_c_f,y_train_f,validation_data=[x_train_v_w_c_f,y_train_f],callbacks=[evaluate_callback,csv_logger],epochs=50,shuffle=False,batch_size=100)
