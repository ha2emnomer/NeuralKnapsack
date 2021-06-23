from pulp import *
import numpy as np
import sys
import math
import argparse
import multiprocessing as mp
import time
def generatebip(n, m, ins, instances = 100 , r=4, type=LpMaximize, cat=LpBinary):
    # n: number of variables
    # m: number of constrains
    r0 = 0
    r1 = 1001

    weights = np.random.randint(1,r1,n)
    #c = np.random.randint(100,12532)
    #c = np.random.uniform(0.2*n,0.3*n)
    #c = np.ceil(c*1e3)
    c = ((ins+1) / float(instances+1)) * sum(weights)
    #c = round(c)
    #c=1250

    #c= 50000
    '''if np.random.randint(0,2):
       c = np.random.randint(r0,r1)
    else:
       c = round((ins+1 / float(instances + 1)) * sum(weights))
    '''
    f = np.random.randint(r0, r1, n)
    #f = weights

    for wi , w in enumerate(weights):
        if w > c:
            c = w
        #f[wi] = round(f[wi],r)
        #weights[wi] = round(weights[wi],r)

        #if w-(r1/10) < 0:
           #f[wi] = np.random.randint(abs(w-(r1/10)),w+(r1/10))
        #else:
        #    f[wi] = np.random.randint(w-(r1/10),w+(r1/10))
        f[wi] = w + (r1-1)/10.0
    '''
    ratios = [f[i] / float(weights[i]) for i in range(len(f))]
    indices = np.argsort(ratios)
    indices = np.flip(indices,0)
    for fi in range(n):
      f[fi] = f[indices[fi]]
      weights[fi] = weights[indices[fi]]
    '''
    #print c
    A = []

    for i in range(m):
        #a = np.random.randint(r0, r1, n)
        A.append([weights, -1, c])
    #indicies = np.argsort(f)
    #f.sort()
    prob = LpProblem("test1", type)
    #prob.solver = solver
    vars = list()
    for i in range(n):
        vars.append(LpVariable("x" + str(i), 0, cat=cat))
    obj = LpAffineExpression([(vars[i], f[i]) for i in range(n)])
    prob += obj
    for i in range(m):
        prob.constraints.update({'c' + str(i): LpConstraint(
            LpAffineExpression([(vars[j], A[i][0][j]) for j in range(n)]), sense=A[i][1],
            rhs=A[i][2])})
        # for i in range(n):
        # prob += LpConstraint(LpAffineExpression(vars[i],1),LpConstraintGE,rhs=0)
    return prob, f, A


# In[3]:

def generateData(samples, min_items=3, max_items=10, r=4 , instances = 100 ,cat=LpBinary,verbose = True):
    x_train_values_weights_cap = np.full([samples, max_items, 3], -1.0)
    y_train_vars = np.zeros((samples, max_items, 2), dtype=np.bool)
    y_objectives = []
    #probs_index = np.random.choice(100, 100, replace=False)
    for i in range(samples):
        items = np.random.randint(min_items, max_items + 1)
        prob, f, A = generatebip(items, 1, i%instances, instances, r, cat=cat)
        try:
            prob.solve()
        except:
            pass
        while LpStatus[prob.status] != 'Optimal':
            prob, f, A = generatebip(items, 1, i%instances, instances, r, cat=cat)
            try:
                prob.solve()
            except:
                pass
        y_objectives.append(value(prob.objective))
        #ratios = [f[fi] / float(A[0][0][fi]) for fi in range(len(f))]
        #indices = np.argsort(ratios)
        #indices = np.flip(indices,0)
        for item in range(items):

            x_train_values_weights_cap[i,item,0] = f[item]
            x_train_values_weights_cap[i,item,1] = A[0][0][item]
            x_train_values_weights_cap[i,item,2] = A[0][2]
        #x_train_values_weights_cap[i,item,2] = A[0][2]

        for v, var in enumerate(prob.variables()):
            y_train_vars[i, int(var.name[1:]), int(var.varValue)] = 1
        #y_train_vars[i, v + 1, 2] = -1
        if verbose:
            sys.stdout.write('Generating ' + str(i) + " / " + str(samples)+"\r")
            sys.stdout.flush()
    #x_train_values_weights_cap /= r

    return x_train_values_weights_cap, y_objectives, y_train_vars


def save_data(path,data ,N,type='train'):
	np.save(path+'/vwc_'+type+'_'+str(N) , data[0])
	np.save(path+'/objectives_'+type+'_'+str(N),data[1]  )
	np.save(path+'/output_'+type+'_'+str(N),data[2] )

parser = argparse.ArgumentParser()
parser.add_argument("-p" ,"--path", type=str, default = 'data',
                    help="path to save data files")
parser.add_argument("-s" ,"--seed", type=int, default = 0,
                    help="seed for random generator")
parser.add_argument("samples", type=int,
                    help="number of samples in dataset")
parser.add_argument("min", type=int, default=2,
                    help="min number of variables")
parser.add_argument("max", type=int, default=10,
                    help="max number of variables")
parser.add_argument("-r" ,"--range", type=int, default = 4,
                    help="decimal percision range")
parser.add_argument("-i" ,"--instances", type=int, default = 100,
                    help="number of instances per capacity")

parser.add_argument("-t" ,"--type", type=str, default = 'train',
                    help="type of dataset train/test")
#start1 = time.time()
args = parser.parse_args()
#pool = mp.Pool(processes=4)
#results = [pool.apply(generateData, args=(args.samples, args.min , args.max , args.range , args.instances,)) ]
#end1 = time.time()
#print end1-start1
#print(results[0])
#print args
np.random.seed(args.seed)
start2 = time.time()
x_train_v_w_c_f, obj_train, y_train_f = generateData(args.samples, args.min , args.max , r = args.range , instances = args.instances)
end2 = time.time()
print x_train_v_w_c_f[0]
#print x_train_v_w_c_f[45]
print 'time taken=' , end2-start2
save_data(args.path , [x_train_v_w_c_f, obj_train, y_train_f], args.max, type= args.type)
