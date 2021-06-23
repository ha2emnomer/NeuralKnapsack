from config_reader import read_config_file
import numpy as np
model_configs = read_config_file('configs/ConfigVisual')
from Utils import saveImage,load_data

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

x_val , obj_val , y_val = load_data([test_path+'vwc_test_'+str(N)+'.npy', test_path+'objectives_test_'+str(N)+'.npy', test_path+'output_test_'+str(N)+'.npy'])
x_val /=Range
x_val = x_val[0:100]
y_val = y_val[0:100]
y_max = np.argmax(y_val,axis=-1) * 255
print y_max
print 'ones=',np.count_nonzero(y_max[:,0])
saveImage(problem_type_test+str(N),np.argmax(y_val,axis=-1))
import numpy as np
import matplotlib.pyplot as plt

zeros = []
ones = []
vars_list = []
for i in range(N):
    ones.append(np.count_nonzero(y_max[:,i])/100.0)
    zeros.append((len(y_max)-np.count_nonzero(y_max[:,i]))/100.0)
    vars_list.append('y'+str(i+1))


# create plot

fig, ax = plt.subplots(figsize=(20, 10))
index = np.arange(N)
bar_width = 0.2
opacity = 0.5

rects1 = plt.bar(index, zeros , bar_width,
alpha=opacity,
color='b',
label='Zeros')

rects2 = plt.bar(index + bar_width, ones, bar_width,
alpha=opacity,
color='g',
label='Ones')

plt.xlabel('Decision Variables')
plt.ylabel('Probablity')
plt.xticks(index + bar_width,vars_list)
plt.legend()

plt.tight_layout()
#plt.show()
plt.savefig('figures/'+problem_type_test+str(N)+'.pdf',bbox_inches='tight')
