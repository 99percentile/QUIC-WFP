import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import gc

def get_data(dir, seq_len=100, test_size = 0.1, num_domains=700, num_traces=333, openworld=False, useLength=True, useTime=True, useDirection=True, useTcp=True, useQuic=True, useBurst=True):
    num_features = sum([useTime, useLength, useDirection, useTcp, useQuic, useBurst])
    X = []
    y = []
    d = os.listdir(dir)
    d.sort()
    for count, filename in enumerate(d):
        if count >= num_traces:
            break
        print(filename)
        if filename.endswith('pickle'):
            try:
                with open(dir+filename, 'rb') as fh:
                    print(dir+filename)
                    array = pickle.load(fh)
                    print(len(array))
            except Exception as e:
                print(e)
                continue
            #print('Finish loading')
            for i in range(len(array)):
                idx = array[i][0]
                #print(idx)
                t = []
                if idx not in y and len(np.unique(y)) == num_domains:
                    continue
               if len(array[i][1]) == 0:
                    continue
               if useLength:
                    lengths = np.array(array[i][1]).reshape((1,-1))
                    lengths = sequence.pad_sequences(lengths, maxlen=seq_len, padding='pre', truncating='pre')
                    lengths = np.array(lengths).reshape((1,-1))
                    t.append(lengths)
                    
                if useTime:
                    times = np.array(array[i][2]).reshape((1,-1))
                    times = sequence.pad_sequences(times, maxlen=seq_len, padding='pre', truncating='pre',dtype=float)
                    times = np.array(times).reshape((1,-1))
                    t.append(times)
                    
                if useDirection:
                    dirs = np.array(array[i][3]).reshape((1,-1))
                    dirs = sequence.pad_sequences(dirs, maxlen=seq_len, padding='pre', truncating='pre')
                    dirs = np.array(dirs).reshape((1,-1))
                    t.append(dirs)
                    
                if useTcp:
                    tcp = np.array(array[i][4]).reshape((1,-1))
                    tcp = sequence.pad_sequences(tcp, maxlen=seq_len, padding='pre', truncating='pre')
                    tcp = np.array(tcp).reshape((1,-1))
                    t.append(tcp)
                
                if useQuic:
                    quic = np.array(array[i][5]).reshape((1,-1))
                    quic = sequence.pad_sequences(quic, maxlen=seq_len, padding='pre', truncating='pre')
                    quic = np.array(quic).reshape((1,-1))
                    if np.sum(quic) < 1:
                        continue
                    t.append(quic)
                
                if useBurst:
                    burst = np.array(array[i][6]).reshape((1,-1))
                    burst = sequence.pad_sequences(burst, maxlen=seq_len, padding='pre', truncating='pre')
                    burst = np.array(burst).reshape((1,-1))
                    t.append(burst)
                
                c = np.column_stack(t)
                
                X.append(c)
                y.append(idx)
    
    X = np.array(X)
    y = np.array(y)
    
    if openworld:
        return X, y
    
    X = X.reshape((-1, num_features, seq_len))
    
    a, b = np.unique(y, return_counts=True)
    a = a.tolist()
    b = b.tolist()
    newX = []
    newy = []
    for smallx, smally in zip(X, y):
        if b[a.index(smally)] < 10:
            continue
        newX.append(smallx)
        newy.append(smally)
    
    newy = np.array(newy)
    newX = np.array(newX)
    X_train, X_test, y_train, y_test = train_test_split(newX, newy, test_size=test_size, stratify=newy)
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1,1))
    
    X_train = X_train.reshape((-1, num_features*seq_len)).reshape((-1, seq_len, num_features), order='F')
    X_test = X_test.reshape((-1, num_features*seq_len)).reshape((-1, seq_len, num_features), order='F')
    
    del X
    del y
    
    gc.collect()
    
    enc = OneHotEncoder(handle_unknown = 'ignore', sparse=False)
    y_train = enc.fit_transform(y_train)
    y_test = enc.transform(y_test)
    
    return X_train, y_train, X_test, y_test

