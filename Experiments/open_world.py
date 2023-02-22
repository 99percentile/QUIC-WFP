import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import gc
import json
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from read_data import get_data

with open('config.json', 'rb') as j:
     config = json.loads(j.read())

seq_len = config['seq_len']
num_domains = config['num_domains']
num_traces = config['num_traces']
useTime = config['useTime']
useLength = config['useLength']
useDirection = config['useDirection']
useTcp = config['useTcp']
useQuic = config['useQuic']
useBurst = config['useBurst']
closed_world_dir = config['closed_world_dir']
open_world_dir = config['open_world_dir']
modelname = config['modelname']

# reading open-world data
with open(open_world_dir+'1-10k.pickle', 'rb') as fh:
    array = pickle.load(fh)

num_features = sum([useTime, useLength, useDirection, useTcp, useQuic, useBurst])
X = []
y = []
for i in range(len(array)):
    idx = array[i][0]
    t = []
    if useLength:
        lengths = np.array(array[i][1]).reshape((1,-1))
        lengths = sequence.pad_sequences(lengths, maxlen=seq_len, padding='post', truncating='post')
        lengths = np.array(lengths).reshape((1,-1))
        t.append(lengths)
        
    if useTime:
        times = np.array(array[i][2]).reshape((1,-1))
        times = sequence.pad_sequences(times, maxlen=seq_len, padding='post', truncating='post',dtype=float)
        times = np.array(times).reshape((1,-1))
        t.append(times)
        
    if useDirection:
        dirs = np.array(array[i][3]).reshape((1,-1))
        dirs = sequence.pad_sequences(dirs, maxlen=seq_len, padding='post', truncating='post')
        dirs = np.array(dirs).reshape((1,-1))
        t.append(dirs)
        
    if useTcp:
        newtcp = list(map(lambda x: -1 if x == 0 else 1, array[i][4]))
        tcp = np.array(newtcp).reshape((1,-1))
        tcp = sequence.pad_sequences(tcp, maxlen=seq_len, padding='post', truncating='post')
        tcp = np.array(tcp).reshape((1,-1))
        t.append(tcp)
    
    if useQuic:
        newquic = list(map(lambda x: -1 if x == 0 else 1, array[i][5]))
        quic = np.array(newquic).reshape((1,-1))
        quic = sequence.pad_sequences(quic, maxlen=seq_len, padding='post', truncating='post')
        quic = np.array(quic).reshape((1,-1))
        t.append(quic)
    
    if useBurst:
        burst = np.array(array[i][6]).reshape((1,-1))
        burst = sequence.pad_sequences(burst, maxlen=seq_len, padding='post', truncating='post')
        burst = np.array(burst).reshape((1,-1))
        t.append(burst)
    
    c = np.column_stack(t)
    
    X.append(c)
    y.append(idx)

X = np.array(X)
y = np.array(y)

num_features = sum([useTime,useLength,useDirection, useTcp, useQuic, useBurst])
X = X.reshape((-1, num_features, seq_len))
X = X.reshape((-1, num_features*seq_len)).reshape((-1, seq_len, num_features), order='F')

X_train, y_train, X_test, y_test = get_data(closed_world_dir, seq_len=seq_len, num_domains=num_domains,
                                                num_traces = num_traces, test_size = 0.1, useLength=useLength,
                                                useTime=useTime, useDirection=useDirection,
                                                useTcp=useTcp, useQuic=useQuic, useBurst=useBurst)


y_test = np.argmax(y_test, axis=1)
X_open, y_open = np.append(X_test, X, axis=0), np.append(y_test, y, axis=0)

del X
del y

X_open = X_open.reshape((-1, seq_len*num_features))
y_open = y_open.reshape((-1, 1))
a = np.hstack((X_open, y_open))
np.random.shuffle(a)
X_open = a[:,:-1]
y_open = a[:,-1]
X_open = X_open.reshape((-1, seq_len, num_features))

# choosing model to use
from tensorflow import keras

lstm = keras.models.load_model('Models/all'+modelname)
df = keras.models.load_model('Models/df'+modelname)
varcnn = keras.models.load_model('Models/varcnn'+modelname)
models = [[lstm, 'all'], [df, 'df'], [varcnn, 'varcnn']]

def get_confusion(proba, y_test, threshold=0.99):
    arg = np.argmax(proba, axis=1)
    val = np.max(proba, axis=1)
    newarg = []
    for i in arg:
        newarg.append(i)
    arg = newarg
    y_pred = []
    for i, j in zip(arg, val):
        if j < threshold:
            y_pred.append(-1)
        else:
            y_pred.append(i)
    
    return get_acc(y_pred, y_test)

corrects = []
def get_acc(gclasses, trueclasses):
    tp = 0
    fp = 0
    wp = 0
    p = 0
    n = 0
    for i in range(0, len(gclasses)):
        if (gclasses[i] != -1):
            #if we guess a postiive class...
            if (gclasses[i] == trueclasses[i]):
                #guess is right
                corrects.append(gclasses[i])
                tp += 1
            elif (trueclasses[i] != -1):
                #guess is wrong, and true class is a positive
                wp += 1
            else:
                #guess is wrong, and true class is a negative
                fp += 1
    
    for i in range(0, len(trueclasses)):
        if (trueclasses[i] == -1):
            n += 1
        else:
            p += 1
    
    return [p, n, tp, wp, fp]


def rprecision(arg, r):
    p, n, tp, wp, fp = arg 
    tpr = tp/p
    wpr = wp/p
    fpr = fp/n
    if tpr == 0 and wpr == 0  and fpr == 0:
        precision = np.nan
    else:
        precision = tpr/(tpr+wpr+r*fpr)
    return tpr, wpr, fpr, precision


#Rvalues = [1,2,3,4,5,6,7,8,9,10, 20, 30, 40, 50]
Rvalues = [10]
n = 10
thresholds = list(np.linspace(0,1,200, endpoint=True))
a = list(reversed(1-np.logspace(-3,0,100, endpoint=True)))[:57]
thresholds.extend(a)
a = list(reversed(1-np.logspace(-5,-3,10)))
thresholds.extend(a)
a = list(reversed(1-np.logspace(-10,-5,50)))
thresholds.extend(a)
thresholds.sort()

r = 10
numTraces = (r+1) * num_domains

precisions = {}
tprs = {}
wprs = {}
fprs = {}

for model, name in models:
    precisions[name] = []
    tprs[name] = []
    wprs[name] = []
    fprs[name] = []

for i in range(20):
    print('round ' + str(i))
    X_open = X_open.reshape((-1, seq_len*num_features))
    y_open = y_open.reshape((-1, 1))
    a = np.hstack((X_open, y_open))
    np.random.shuffle(a)
    X_open = a[:,:-1]
    y_open = a[:,-1]
    X_open = X_open.reshape((-1, seq_len, num_features))
    
    newX = []
    newy = []
    for x, y in zip(X_open, y_open):
        if y in newy:
            continue
        if y < num_domains:
            newX.append(x)
            newy.append(y)

    count = []
    for x, y in zip(X_open, y_open):
        if y in count:
            continue
        if y >= num_domains and y < numTraces:
            newX.append(x)
            newy.append(-1)
            count.append(y)

    newX = np.array(newX)
    newy = np.array(newy)

    for model, name in models:
        print(name)
        if name == 'all':
            proba = model.predict([newX[:,:,:-1], newX[:,:,-1]])
        
        if name == 'varcnn':
            proba = model.predict([newX[:,:,1], newX[:,:,2]])
        
        if name == 'df':
            proba = model.predict(newX[:,:,2])
        
        for threshold in thresholds:
            for r in Rvalues:
                tpr, wpr, fpr, cal = rprecision(get_confusion(proba, newy, threshold), r)
                precisions[name].append(cal)
                tprs[name].append(tpr)
                wprs[name].append(wpr)
                fprs[name].append(fpr)
        
        
        #print('tprs=', tprs)
        #print('wprs=', wprs)
        #print('fprs=', fprs)
        #print('precisions=', precisions)
        
        #np.save('Results/tprsopen'+modelname+name+'.npy', tprs)
        #np.save('Results/wprsopen'+modelname+name+'.npy', wprs)
        #np.save('Results/fprsopen'+modelname+name+'.npy', fprs)
        #np.save('Results/precisionsopen'+modelname+name+'.npy', precisions)

with open('Results/precisions'+modelname+'.pickle', 'wb') as handle:
    pickle.dump(precisions, handle)

with open('Results/tprs'+modelname+'.pickle', 'wb') as handle:
    pickle.dump(tprs, handle)

with open('Results/fprs'+modelname+'.pickle', 'wb') as handle:
    pickle.dump(fprs, handle)

with open('Results/wprs'+modelname+'.pickle', 'wb') as handle:
    pickle.dump(wprs, handle)