from config_reader import read_config_file
from CSVDataFrame import CSVDataFrame
import matplotlib.pyplot as plt
import math
model_configs = read_config_file('configs/plotConfig')
#config
hidden =  int(model_configs['HIDDEN'])
N = model_configs['N']
mtype = model_configs['TYPE']
Range = int(model_configs['RANGE'])
mem_layers = int(model_configs['MEM_LAYERS'])
dropout = model_configs['DROPOUT']
problem_type = model_configs['PROBLEM_TYPE']
problem_type_test = model_configs['PROBLEM_TYPE_TEST']
size = int(model_configs['TEST_SIZE'])
mname = None
if mtype == 'cnn':
    mname = 'Neuroknapsack-CNN-' +problem_type+'-'+str(Range)+'-'+str(N)+'-'+str(hidden)+'-'+str(mem_layers)+'-'+str(dropout[0])+str(dropout[1])+str(dropout[2])
elif mtype == 'gru-memory':
    mname = 'Neuroknapsack-gru-memory-' +problem_type+'-'+str(Range)+'-'+str(N)+'-'+str(hidden)+'-'+str(mem_layers)+'-'+str(dropout[0])+str(dropout[1])+str(dropout[2])
elif mtype == 'seq2seq':
    mname = 'Seq2seq-' +problem_type+'-'+str(Range)+'-'+str(N)+'-'+str(hidden)+'-'+str(mem_layers)+'-'+str(dropout[0])+str(dropout[1])+str(dropout[2])
elif mtype == 'gru':
     mname = 'GRU-' + str(Range)+'-'+str(N)+'-'+str(hidden)
else:
    mname = 'Neuroknapsack-DENSE-'+problem_type+'-'+str(Range)+'-'+str(N)+'-'+str(hidden)+'-'+str(mem_layers)+'-'+str(dropout[0])+str(dropout[1])+str(dropout[2])
print mname
test_log = CSVDataFrame('logs/test-logs/'+mname+'.csv')
train_log = CSVDataFrame('logs/training-logs/'+mname+'.log')
test_log.ReadCSV()
train_log.ReadCSV()
epochs = [int(e) for s in train_log.selectbycol(keys=['epoch']) for e in s]
t_loss = [float(e) for s in train_log.selectbycol(keys=['loss']) for e in s]
v_loss = [float(e) for s in train_log.selectbycol(keys=['val_loss']) for e in s]

t_loss_ema = [t_loss[0]]
v_loss_ema = [v_loss[0]]
for i in range(1,len(epochs),1):
    t_loss_ema.append(0.7*t_loss[i]+(1-0.7)*t_loss_ema[-1])
    v_loss_ema.append(0.7*v_loss[i]+(1-0.7)*v_loss_ema[-1])
loss_fig  = plt.figure(1)
plt.plot(epochs, t_loss_ema, 'b', epochs, v_loss_ema, 'r')
loss_fig.show()
infeas_problems = [float(e)/size for s in test_log.selectbycol(keys=['% infeas problems']) for e in s]
model_accuracy = [float(e)/size for s in test_log.selectbycol(keys=['model accuracy']) for e in s]
infeas_problems_fig  = plt.figure(2)
plt.plot(epochs, infeas_problems, 'b--',epochs, model_accuracy, 'k--')
infeas_problems_fig.show()
raw_input()
