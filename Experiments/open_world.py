from lstmqfp import lstm_qfp
from read_data import get_data
import json
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

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

lstm = lstm_qfp(seq_len, num_domains, useTime, useLength, useDirection, useTcp, useQuic, useBurst)
    
X_train, y_train, X_test, y_test = get_data(closed_world_dir, seq_len=seq_len, num_domains=num_domains,
                                            num_traces = num_traces, test_size = 0.1, useLength=useLength,
                                            useTime=useTime, useDirection=useDirection,
                                            useTcp=useTcp, useQuic=useQuic, useBurst=useBurst)

lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_accuracy', patience = 10, verbose=1)
if useBurst:
    history= lstm.fit([X_train[:,:,:-1], X_train[:,:,-1]], y_train, validation_split=0.15, epochs=100, batch_size=32, use_multiprocessing=True, callbacks=[early_stopping], workers=20)
else:
    history= lstm.fit(X_train, y_train, validation_split=0.15, epochs=100, batch_size=32, use_multiprocessing=True, callbacks=[early_stopping], workers=20)

print("Test Results")

if useBurst:
    test_acc = lstm.evaluate(x=[X_test[:,:,:-1],X_test[:,:,-1]], y=y_test)
else:
    test_acc = lstm.evaluate(X_test, y=y_test)

print(test_acc, flush=True)

X_open, y_open = get_data(open_world_dir, seq_len=seq_len, num_domains=num_domains,
                                            num_traces = 1, test_size = 0.1, openworld = True, useLength=useLength,
                                            useTime=useTime, useDirection=useDirection,
                                            useTcp=useTcp, useQuic=useQuic, useBurst=useBurst)



def get_confusion(model, X_test, y_test, threshold=0.99):
    # varcnn
    #proba = model.predict(x=[X_test[:,:,1], X_test[:,:,2]])
    
    # lstm-qfp
    proba = model.predict([X_test[:,:,:-1],X_test[:,:,-1]])
    
    # tcp only
    #proba = model.predict(X_test)
    
    # lstm-qfp burst
    #proba = model.predict([X_test[:,:,:5], X_test[:,:,5:8], X_test[:,:,8:]])
    
    # df
    #proba = model.predict(X_test[:,:,2])
    
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
thresholds = np.linspace(0,1,200, endpoint=False)

for threshold in thresholds:
    for r in Rvalues:
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
        
        tpr, wpr, fpr, cal = rprecision(get_confusion(lstm, newX, newy, threshold), r)
        precisions.append(cal)
        tprs.append(tpr)
        wprs.append(wpr)
        fprs.append(fpr)


print('tprs=', tprs)
print('wprs=', wprs)
print('fprs=', fprs)
print('precisions=', precisions)

