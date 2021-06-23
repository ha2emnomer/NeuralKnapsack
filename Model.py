import Models
from Utils import checkcapacity,getcost,problems_accuracy,beam_search_decoder
from Greedy import greedyKnapsack
from keras.utils import to_categorical
import numpy as np
import sys
class Model(object):
    def __init__(self,mtype,problem_type,n,range,hidden, mem_layers = 1 , dropout_memory = 0.0,dropout_encoder=0.0,dropout_decoder=0.0):
        self.model_type = mtype
        self.hidden = hidden
        self.mem_layers = mem_layers
        self.dropout_memory = dropout_memory
        self.dropout_encoder = dropout_encoder
        self.dropout_decoder = dropout_decoder
        self.encoder  = None
        self.decoder = None
        self.model = None
        self.model_cos = None
        self.Range = range
        self.N = n
        self.name = None
        self.problem_type = problem_type
    def build(self):
        if self.model_type == 'cnn':
            self.model,self.encoder,self.decoder,self.model_cos = Models.neuroknapsack(self.hidden,True,False,self.mem_layers,self.dropout_memory,
            self.dropout_encoder,self.dropout_decoder)
        elif self.model_type == 'gru-memory':
            self.model,self.encoder,self.decoder,self.model_cos = Models.neuroknapsack(self.hidden,False,True,self.mem_layers,self.dropout_memory,
            self.dropout_encoder,self.dropout_decoder)
        elif self.model_type == 'gru':
            self.model = Models.grumodel(self.hidden,self.dropout_encoder)
        elif self.model_type == 'seq2seq':
            self.model,self.encoder,self.decoder = Models.seq2seq(self.hidden,
            self.dropout_encoder,self.dropout_decoder)
        else:
            self.model,self.encoder,self.decoder,self.model_cos=Models.neuroknapsack(self.hidden,False,False,self.mem_layers,self.dropout_memory,
            self.dropout_encoder,self.dropout_decoder)
    def updaterange(self,range):
        self.Range = range
    def update(self,N):
        self.N = N
        if self.model_type == 'cnn':
            self.name = 'Neuroknapsack-CNN-' +self.problem_type+'-'+str(self.Range)+'-'+str(self.N)+'-'+str(self.hidden)+'-'+str(self.mem_layers)+'-'+str(self.dropout_memory)+str(self.dropout_encoder)+str(self.dropout_decoder)
        elif self.model_type == 'gru-memory':
            self.name = 'Neuroknapsack-gru-memory-' +self.problem_type+'-'+str(self.Range)+'-'+str(self.N)+'-'+str(self.hidden)+'-'+str(self.mem_layers)+'-'+str(self.dropout_memory)+str(self.dropout_encoder)+str(self.dropout_decoder)
        elif self.model_type == 'seq2seq':
            self.name = 'Seq2seq-' +self.problem_type+'-'+str(self.Range)+'-'+str(self.N)+'-'+str(self.hidden)+'-'+str(self.mem_layers)+'-'+str(self.dropout_memory)+str(self.dropout_encoder)+str(self.dropout_decoder)
        elif self.model_type == 'gru':
             self.name = 'GRU-' + str(self.Range)+'-'+str(self.N)+'-'+str(self.hidden)
        elif self.model_type == 'DENSE':
            self.name = 'Neuroknapsack-DENSE-'+self.problem_type+'-'+str(self.Range)+'-'+str(self.N)+'-'+str(self.hidden)+'-'+str(self.mem_layers)+'-'+str(self.dropout_memory)+str(self.dropout_encoder)+str(self.dropout_decoder)
        else:
            self.name = 'Neuroknapsack-'+self.model_type+'-'+self.problem_type+'-'+str(self.Range)+'-'+str(self.N)+'-'+str(self.hidden)+'-'+str(self.mem_layers)+'-'+str(self.dropout_memory)+str(self.dropout_encoder)+str(self.dropout_decoder)
    def dividetest(self,x_val,y_val,obj_val):

        y_p = np.zeros((self.N,2))
        zeros = np.zeros((1,self.N,2))
        acc = 0
        acc_greedy = 0
        model_feas = 0
        feas_greedy = 0
        solver_costs = 0
        model_costs = 0
        greedy_costs = 0
        total_correct_variables = 0
        infeas = 0
        test_set = x_val.shape[0]
        y_final = np.zeros((test_set,self.N,2))

        for _i in range(test_set):


            weights = []
            values = []
            #for t in range(self.N):
                #weights.append(x_val[_i, t, 1]*self.Range)
                #values.append(x_val[_i, t, 0]*self.Range)

            weights  = x_val[_i,0:self.N,1] * self.Range
            values = x_val[_i,0:self.N,0] * self.Range
            #solver optimial
            solver_costs += obj_val[_i]
            #greedyKnapsack
            y = greedyKnapsack(values, weights, x_val[_i, 0, 2]*self.Range)
            es_g , reward_g = problems_accuracy(y_val[_i], y, self.N)
            if es_g == 1:
                acc_greedy += 1
            if checkcapacity(weights, y, x_val[_i, 0, 2]*self.Range):
                feas_greedy += 1
            greedy_costs += getcost(values, y)
            for t in range(0,self.N,20):
                y_p_, rt = Models.decode_sequence(x_val[_i:_i+1,t:t+20],20,self.encoder,self.decoder)
                y_p[t:t+20] = y_p_
            if checkcapacity(weights, y_p, x_val[_i, 0, 2]*self.Range):
                model_costs+= getcost(values,y_p)
                model_feas+=1
                es , reward = problems_accuracy(y_val[_i], y_p, self.N)
                if es == 1:
                    acc += 1
                total_correct_variables += reward
            else:
                infeas+=1
                #continue
                indices = np.argsort(weights)
                indices = np.flip(indices,0)
                for w,v in enumerate(indices):
                    #print _i, w
                    if np.argmax(y_p[w]) == 1:
                        y_p[w,0] = 1
                        y_p[w,1] = 0
                        if checkcapacity(weights, y_p,x_val[_i, 0, 2]*self.Range):
                            model_costs+= getcost(values,y_p)
                            model_feas+=1
                            es , reward = problems_accuracy(y_val[_i], y_p, self.N)
                            #if es == 1:
                            #    acc += 1
                            #total_correct_variables += reward
                            break
            y_final[_i] = y_p

        model_approx_ratio = max(solver_costs / model_costs, model_costs / solver_costs)
        greedy_approx_ratio = max(solver_costs / greedy_costs, greedy_costs / solver_costs)
        print '\r'
        print ('% infeasible problems:' , infeas / float(test_set))
        print ('total number of correct variables:' , total_correct_variables / float(test_set*self.N))
        print ('number of feasible sols (greedy):', feas_greedy/ float(test_set))
        print ('greedy costs:', greedy_costs)
        print ('solver costs:', solver_costs)
        print ('Neuroknapsack costs:', model_costs)
        print ('accuracy (Neuroknapsack):', acc / float(model_feas))
        print ('accuracy (greedy):', acc_greedy / float(model_feas))
        print ('number of feasible sols (Neuroknapsack):', model_feas / float(test_set))
        print ('Approx. ratio for (Neuroknapsack):', model_approx_ratio)
        print ('Approx. ratio for greedy algorithm:', greedy_approx_ratio)
        return infeas, acc, acc_greedy, model_approx_ratio,greedy_approx_ratio
    def test(self,x_val,y_val,obj_val):

        y_p = np.zeros((self.N,2))
        zeros = np.zeros((1,self.N,2))
        acc = 0
        acc_greedy = 0
        model_feas = 0
        feas_greedy = 0
        solver_costs = 0
        model_costs = 0
        greedy_costs = 0
        total_correct_variables = 0
        infeas = 0
        test_set = x_val.shape[0]
        y_final = np.zeros((test_set,self.N,2))


        for _i in range(test_set):


            weights = []
            values = []
            #for t in range(self.N):
                #weights.append(x_val[_i, t, 1]*self.Range)
                #values.append(x_val[_i, t, 0]*self.Range)

            weights  = x_val[_i,0:self.N,1] * self.Range
            values = x_val[_i,0:self.N,0] * self.Range
            #solver optimial
            solver_costs += obj_val[_i]
            #greedyKnapsack
            y = greedyKnapsack(values, weights, x_val[_i, 0, 2]*self.Range)
            es_g , reward_g = problems_accuracy(y_val[_i], y, self.N)
            if es_g == 1:
                acc_greedy += 1
            if checkcapacity(weights, y, x_val[_i, 0, 2]*self.Range):
                feas_greedy += 1
            greedy_costs += getcost(values, y)


            if self.model_type == 'gru':
                y_p = self.model.predict(x_val[_i:_i+1])[0]
            elif self.model_type == 'seq2seq':
                y_p = Models.decode_sequence_seq2seq(x_val[_i:_i+1],self.N,self.encoder,self.decoder)
            elif self.model_type == 'teacher':
                y_p, rt = Models.decode_sequence_teacher(x_val[_i:_i+1],y_val[_i:_i+1],self.N,self.encoder,self.decoder)
            elif self.model_type == 'st':
                y_p, rt = Models.decode_sequence_teacher(x_val[_i:_i+1],y_val[_i:_i+1],self.N,self.encoder,self.decoder)
            else:
                 y_p, rt = Models.decode_sequence(x_val[_i:_i+1],self.N,self.encoder,self.decoder)
            if checkcapacity(weights, y_p, x_val[_i, 0, 2]*self.Range):
                model_costs+= getcost(values,y_p)
                model_feas+=1
                es , reward = problems_accuracy(y_val[_i], y_p, self.N)
                if es == 1:
                    acc += 1
                total_correct_variables += reward
            else:
                infeas+=1
                #continue
                indices = np.argsort(weights)
                indices = np.flip(indices,0)
                for w,v in enumerate(indices):
                    #print _i, w
                    if np.argmax(y_p[w]) == 1:
                        y_p[w,0] = 1
                        y_p[w,1] = 0
                        if checkcapacity(weights, y_p,x_val[_i, 0, 2]*self.Range):
                            model_costs+= getcost(values,y_p)
                            model_feas+=1
                            es , reward = problems_accuracy(y_val[_i], y_p, self.N)
                            #if es == 1:
                            #    acc += 1
                            #total_correct_variables += reward
                            break
            y_final[_i] = y_p
            sys.stdout.write('solved ' + str(_i+1) + " / " + str(test_set)+"\r")
            sys.stdout.flush()

        model_approx_ratio = max(solver_costs / model_costs, model_costs / solver_costs)
        greedy_approx_ratio = max(solver_costs / greedy_costs, greedy_costs / solver_costs)
        print '\r'
        print ('% infeasible problems:' , infeas / float(test_set))
        print ('total number of correct variables:' , total_correct_variables / float(test_set*self.N))
        print ('number of feasible sols (greedy):', feas_greedy/ float(test_set))
        print ('greedy costs:', greedy_costs)
        print ('solver costs:', solver_costs)
        print ('Neuroknapsack costs:', model_costs)
        print ('accuracy (Neuroknapsack):', acc / float(model_feas))
        print ('accuracy (greedy):', acc_greedy / float(model_feas))
        print ('number of feasible sols (Neuroknapsack):', model_feas / float(test_set))
        print ('Approx. ratio for (Neuroknapsack):', model_approx_ratio)
        print ('Approx. ratio for greedy algorithm:', greedy_approx_ratio)
        return infeas, acc, acc_greedy, model_approx_ratio,greedy_approx_ratio
