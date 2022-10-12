from lstmqfp import lstm_qfp
from read_data import get_data
import json
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pickle

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

if useBurst:
    proba = lstm.predict(x=[X_test[:,:,:-1],X_test[:,:,-1]])
else:
    proba = lstm.predict(X_test)

arg = np.argmax(proba, axis=1)
val = np.max(proba, axis=1)
newarg = []
for i in arg:
    newarg.append(i)

newval = []
wrongtrue = []
classifiedas = []
for k, (i, j) in enumerate(zip(newarg, np.argmax(y_test, axis=1))):
    if i != j:
        newval.append(val[k])
        wrongtrue.append(j)
        classifiedas.append(i)

with open('wrongtrue5k.data', 'wb') as f:
    pickle.dump(wrongtrue, f)

with open('classifiedas5k.data', 'wb') as f:
    pickle.dump(classifiedas, f)

