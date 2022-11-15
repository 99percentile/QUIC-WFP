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

with open('config.json', 'rb') as j:
     config = json.loads(j.read())

seq_len = config['seq_len']
num_domains = config['num_domains']
num_traces = config['num_traces']
useTime = True
useLength = True
useDirection = True
useTcp = True
useQuic = True
useBurst = True
closed_world_dir = config['closed_world_dir']
open_world_dir = config['open_world_dir']


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

X_open, y_open = X, y

del X
del y

# choosing model to use
from tensorflow import keras

lstm = keras.models.load_model('Models/all100')
#df = keras.models.load_model('Models/df100')
#varcnn = keras.models.load_model('Models/varcnn100')
model = lstm

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
Rvalues = [20]
precisions = []
tprs = []
wprs = []
fprs = []
n = 10
thresholds = list(np.linspace(0,1,200, endpoint=True))
a = list(reversed(1-np.logspace(-3,0,100, endpoint=True)))[:57]
thresholds.extend(a)
a = list(reversed(1-np.logspace(-5,-3,10)))
thresholds.extend(a)
thresholds.sort()

r = 20
newX = []
newy = []
numTraces = r * num_domains
count = 0
for x, y in zip(X_open, y_open):
    if y < num_domains:
        newX.append(x)
        newy.append(y)
        count += 1
        if count >= num_domains:
            break

count = 0
for x, y in zip(X_open, y_open):
    if y >= num_domains:
        newX.append(x)
        newy.append(-1)
        count += 1
        if count >= numTraces:
            break

newX = np.array(newX)
newy = np.array(newy)

num_features = sum([useTime,useLength,useDirection, useTcp, useQuic, useBurst])
newX = newX.reshape((-1, num_features, seq_len))
newX = newX.reshape((-1, num_features*seq_len)).reshape((-1, seq_len, num_features), order='F')

#lstm
proba = model.predict([newX[:,:,:-1], newX[:,:,-1]])

#varcnn
#proba = model.predict([newX[:,:,1], newX[:,:,2]])

#df
#proba = model.predict(newX[:,:,2])

for threshold in thresholds:
    for r in Rvalues:        
        tpr, wpr, fpr, cal = rprecision(get_confusion(proba, newy, threshold), r)
        precisions.append(cal)
        tprs.append(tpr)
        wprs.append(wpr)
        fprs.append(fpr)


print('tprs=', tprs)
print('wprs=', wprs)
print('fprs=', fprs)
print('precisions=', precisions)

np.save('Results/tprsall100.npy', tprs)
np.save('Results/wprsall100.npy', wprs)
np.save('Results/fprsall100.npy', fprs)
np.save('Results/precisionsall100.npy', precisions)
