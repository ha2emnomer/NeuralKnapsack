from keras.callbacks import Callback
from CSVDataFrame import CSVDataFrame
import numpy as np
class EvaluateCallBack(Callback):
    def __init__(self,model,encoder,model_name,x_val,y_val,obj_val,test_fn,pool = None,max_wait=5):
        self.x_val = x_val
        self.y_val = y_val
        self.obj_val = obj_val
        self.pool = pool
        self.test_func = test_fn
        self.metrics = []
        self.w_metrics = [1000.0]
        self.wait = 0
        self.max_wait = max_wait
        self.model_name = model_name
        self.model  = model
        self.encoder = encoder
        self.frame = CSVDataFrame()
        self.frame.setheader(['Epoch','% infeas problems', 'model accuracy', 'greedy accuracy', 'model_approx_ratio','greedy_approx_ratio'])
    '''
    def on_batch_begin(self,batch,log):
        if bool != None:
            if batch // 5:
                m = np.zeros((100,32))
                m = self.encoder.predict(pool)
                self.model.get_layer('Memory').set_weights([m])
                self.model.get_layer('mem').set_weights([np.transpose(m)])
     '''



    def on_epoch_end(self,epoch,log):

        infeas, acc, acc_greedy, model_approx_ratio,greedy_approx_ratio = self.test_func(self.x_val,self.y_val,self.obj_val)
        if model_approx_ratio >= min(self.w_metrics):
            if self.wait >= self.max_wait:
                print 'training stopped' , self.wait , self.max_wait
                self.model.stop_training = True
                self.wait = 0

            else:
                self.wait+=1
                #self.metrics.append(infeas)

        else:
            self.wait = 0
            print 'model saved...'
            print model_approx_ratio , min(self.w_metrics)
            self.model.save('.models/'+self.model_name+'.hdf5')

        self.metrics.append([epoch+1,infeas, acc, acc_greedy, model_approx_ratio,greedy_approx_ratio])
        self.w_metrics.append(model_approx_ratio)
        self.frame.PassDataFrame(self.metrics)


    def on_train_end(self,logs):
        self.frame.save('logs/test-logs/'+self.model_name+'.csv')
