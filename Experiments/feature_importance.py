from lstmqfp import lstm_qfp
from read_data import get_data
import json
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)



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

X_train, y_train, X_test, y_test = get_data(closed_world_dir, seq_len=seq_len, num_domains=num_domains,
                                                num_traces = num_traces, test_size = 0.1, useLength=useLength,
                                                useTime=useTime, useDirection=useDirection,
                                                useTcp=useTcp, useQuic=useQuic, useBurst=useBurst)

results = []
features = [False for i in range(6)]

for f in range(5, 6):
    features[f] = True
    for _ in range(10):
        lstm = lstm_qfp(seq_len, num_domains, features[0], features[1], features[2], features[3], features[5], features[4])
        
        early_stopping = EarlyStopping(monitor='val_accuracy', patience = 10, verbose=1)
        #X_train = X_train[:,:,f]
        #X_test = X_test[:,:,f]
        lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history= lstm.fit(X_train[:,:,f], y_train, validation_split=0.15, epochs=100, batch_size=32, use_multiprocessing=True, callbacks=[early_stopping], workers=20)
        test_acc = lstm.evaluate(X_test[:,:,f], y=y_test)
        
        results.append(test_acc)
        print('lstm:', test_acc)
        lstm.save('Models/'+str(f))
    
    features[f] = False


print(results, flush=True)

