from lstmqfp import lstm_qfp
from read_data import get_data
import json
from tensorflow.keras.callbacks import EarlyStopping

with open('config.json', 'rb') as j:
     config = json.loads(j.read())

seq_len = 5000
num_domains = config['num_domains']
num_traces = config['num_traces']
useTime = config['useTime']
useLength = config['useLength']
useDirection = config['useDirection']
useTcp = config['useTcp']
useQuic = config['useQuic']
useBurst = config['useBurst']
closed_world_dir = config['closed_world_dir']

sequence_lengths = [10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000]

results = []

X_train, y_train, X_test, y_test = get_data(closed_world_dir, seq_len=seq_len, num_domains=num_domains,
                                                num_traces = num_traces, test_size = 0.1, useLength=useLength,
                                                useTime=useTime, useDirection=useDirection,
                                                useTcp=useTcp, useQuic=useQuic, useBurst=useBurst)

for new_seqlen in sequence_lengths:
    lstm = lstm_qfp(seq_len, num_domains, useTime, useLength, useDirection, useTcp, useQuic, useBurst)
    
    X_train, X_test = X_train[:new_seqlen,:,:], X_test[:new_seqlen,:,:]
    
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
    results.append(test_acc)
    print(test_acc, flush=True)


print(results)