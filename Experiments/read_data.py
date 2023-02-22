import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import gc

def pad_and_truncate(seq, maxlen):
    l = len(seq)
    if l > maxlen:
        seq = seq[:maxlen]
    pad = maxlen-len(seq)
    return np.append(np.zeros(pad), seq).reshape((1, -1))

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
                    lengths = np.array(array[i][1])
                    lengths = pad_and_truncate(lengths, seq_len)
                    t.append(lengths)
                    
                if useTime:
                    times = np.array(array[i][2])
                    times = pad_and_truncate(times, seq_len)
                    t.append(times)
                    
                if useDirection:
                    dirs = np.array(array[i][3])
                    dirs = pad_and_truncate(dirs, seq_len)
                    t.append(dirs)
                    
                if useTcp:
                    tcp = np.array(array[i][4])
                    tcp = pad_and_truncate(tcp, seq_len)
                    t.append(tcp)
                
                if useQuic:
                    quic = np.array(array[i][5])
                    quic = pad_and_truncate(quic, seq_len)
                    t.append(quic)
                
                if useBurst:
                    burst = np.array(array[i][6])
                    burst = pad_and_truncate(burst, seq_len)
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
    X_train, X_test, y_train, y_test = train_test_split(newX, newy, test_size=test_size, stratify=newy, random_state=42)
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

