from config_reader import read_config_file
import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop, Adam, SGD,Adadelta
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping,CSVLogger,LearningRateScheduler, RemoteMonitor
from Model import Model
from Utils import load_data,saveImage
from EvaluateCallBack import EvaluateCallBack
import keras.backend as k
np.random.seed(50)
model_configs = read_config_file('configs/testConfig')


#config
hidden =  int(model_configs['HIDDEN'])
N = model_configs['N']
mtype = model_configs['TYPE']
Range = float(model_configs['RANGE'])
mem_layers = int(model_configs['MEM_LAYERS'])
dropout = model_configs['DROPOUT']
problem_type = model_configs['PROBLEM_TYPE']
problem_type_test = model_configs['PROBLEM_TYPE_TEST']
max_N = int(model_configs['MAX_N'])
train_path = 'data/train/'+problem_type+'/Range-'+str(model_configs['RANGE_TEST'])+'/'
test_path = 'data/test/'+problem_type_test+'/Range-'+str(model_configs['RANGE_TEST'])+'/'
optimizer = Adam()
print test_path

#create model
model = Model(mtype,problem_type,N[0],model_configs['RANGE'],hidden,mem_layers,dropout[0],dropout[1],dropout[2])

#load & normalize data
x_vwc_f = []
y_tm1_f = []
for j,n in enumerate(N):
    print j,n
    #x_val , obj_val , y_val = load_data([test_path+'vwc_pool_'+str(n)+'.npy', test_path+'objectives_pool_'+str(n)+'.npy', test_path+'output_pool_'+str(n)+'.npy'])
    x_val , obj_val , y_val = load_data([test_path+'vwc_test_'+str(n)+'.npy', test_path+'objectives_test_'+str(n)+'.npy', test_path+'output_test_'+str(n)+'.npy'])
    print 'max_cap=',np.amax(x_val[:,:,2])
    print 'test size = ', x_val.shape[0]
    print  x_val
    #x_val /= float(model_configs['RANGE_TEST'])
    x_val /= float(Range)
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
    elif mtype == 'GAN':
        ST = GAN(hidden,dropout[0])
        stmodel ,student ,teacher , student_encoder,student_decoder,teacher_encoder, teacher_decoder = ST.build(mem_layers)
        student.summary()
        teacher.summary()

        #assign encoder and decoder models
        model.encoder = teacher_encoder
        model.decoder = teacher_decoder
        model.model_type ='teacher'
        y_tm1 = np.zeros((train_instances,n,n,2))
        for t in range(n):
            y_tm1[:, t, 0:t, :] = y[:, 0:t, :]
        evaluate_callback_teacher = EvaluateCallBack(teacher,model.name+'-teacher',x_val,y_val,obj_val,model.test,5)
        csv_logger = CSVLogger('logs/training-logs/'+model.name+'-teacher.log')
        teacher.fit([x_vwc,y,x_vwc,y_tm1],[y], shuffle=False, validation_split=0.5,epochs=1,batch_size=100,callbacks=[csv_logger,evaluate_callback_teacher])
        teacher.trainable = False
        stmodel.compile(optimizer=stmodel.optimizer, loss= stmodel.loss, metrics= stmodel.metrics)
        stmodel.summary()
        model.encoder = student_encoder
        model.decoder = student_decoder
        model.model_type ='student'
        evaluate_callback_student = EvaluateCallBack(student,model.name+'-student',x_val,y_val,obj_val,model.test,max_wait)
        csv_logger = CSVLogger('logs/training-logs/'+model.name+'-student.log')
        stmodel.fit([x_vwc,y,x_vwc,y_tm1],[y], shuffle=False, validation_split=0.1,epochs=100,batch_size=100,callbacks=[csv_logger,evaluate_callback_student])
        model.model = student
    else:
        if j == 0:
            model.update(max_N)
            model.build()
            print model.name
            model.model.summary()
            model.model.load_weights('.models/'+model.name+'.hdf5')
        model.update(n)
        model.Range = float(model_configs['RANGE_TEST'])
        model.test(x_val,y_val,obj_val)

        '''
        y_max = np.argmax(y_final,axis=-1)
        print y_max[0]
        print 'ones=',np.count_nonzero(y_max[:,0])
        saveImage('results/'+problem_type_test+'/'+str(n)+'_predicted',np.argmax(y_final[0:100],axis=-1))
        y_max = np.argmax(y_val,axis=-1)
        print 'ones=',np.count_nonzero(y_max[:,0])
        saveImage('results/'+problem_type_test+'/'+str(n)+'_optimal',np.argmax(y_val[0:100],axis=-1))
        '''
