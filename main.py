from config_reader import read_config_file
import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop, Adam, SGD,Adadelta
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping,CSVLogger,LearningRateScheduler, RemoteMonitor
from Model import Model
from Utils import load_data
from EvaluateCallBack import EvaluateCallBack
import keras.backend as k
np.random.seed(50)
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

model_configs = read_config_file('configs/Config')


#config
hidden =  int(model_configs['HIDDEN'])
N = model_configs['N']
mtype = model_configs['TYPE']
Range = float(model_configs['RANGE'])
mem_layers = int(model_configs['MEM_LAYERS'])
dropout = model_configs['DROPOUT']
problem_type = model_configs['PROBLEM_TYPE']
problem_type_test = model_configs['PROBLEM_TYPE_TEST']
max_wait = int(model_configs['MAX_WAIT'])
train_path = 'data/train/'+problem_type+'/Range-'+str(model_configs['RANGE'])+'/'
test_path = 'data/test/'+problem_type_test+'/Range-'+str(model_configs['RANGE'])+'/'



#create model
model = Model(mtype,problem_type,N[0],model_configs['RANGE'],hidden,mem_layers,dropout[0],dropout[1],dropout[2])
#model.Range = 1
#load & normalize data
x_vwc_f = []
y_tm1_f = []
for j,n in enumerate(N):
    model.update(n)
    print model.name
    x_vwc ,obj_train, y = load_data([train_path+'vwc_train_'+str(n)+'.npy', train_path+'objectives_train_'+str(n)+'.npy',train_path+'output_train_'+str(n)+'.npy'])
    x_vwc /= Range
    train_instances = x_vwc.shape[0]
    x_val , obj_val , y_val = load_data([test_path+'vwc_test_'+str(n)+'.npy', test_path+'objectives_test_'+str(n)+'.npy', test_path+'output_test_'+str(n)+'.npy'])
    x_val /=Range
    #x_pool , _,_ = load_data([test_path+'vwc_pool_'+str(n)+'.npy', test_path+'objectives_pool_'+str(n)+'.npy', test_path+'output_pool_'+str(n)+'.npy'])
    print x_val[0]
    print x_vwc.shape[0]
    evaluate_callback = EvaluateCallBack(model.model,model.encoder,model.name,x_val,y_val,obj_val,model.test,max_wait=max_wait)
    csv_logger = CSVLogger('logs/training-logs/'+model.name+'.log')

    if mtype == 'gru':
        model.build()
        model.model.summary()
        model.model.fit(x_vwc,y,validation_split=0.1,callbacks=[evaluate_callback,csv_logger],epochs=100,shuffle=False,batch_size=100)
    elif mtype == 'seq2seq':
        model.build()
        model.model.summary()
        y_tm1 = np.zeros((train_instances,n,2))
        for t in range(1,n):
            y_tm1[:, t, :] = y[:, t-1, :]
        print y_tm1[0]
        model.model.fit([x_vwc,x_vwc,y_tm1],[y], shuffle= False, validation_split=0.1,epochs=100,batch_size=100,callbacks=[csv_logger,evaluate_callback])
    else:
        model.build()
        model.model.summary()
        y_tm1 = np.zeros((train_instances,n,n,2))
        for t in range(n):
            y_tm1[:, t, 0:t, :] = y[:, 0:t, :]
        if j !=0:
           model.model.fit([x_vwc,y_tm1],[y], shuffle=False, validation_split=0.25,epochs=100,batch_size=100,callbacks=[csv_logger,evaluate_callback])


    #model.model.load_weights('.models/'+model.name+'.hdf5')
