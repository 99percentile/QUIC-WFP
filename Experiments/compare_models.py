from lstmqfp import lstm_qfp
from df import DFNet
from varcnn import varcnn
from read_data import get_data
import json
from tensorflow.keras.callbacks import EarlyStopping

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

lstm = lstm_qfp(seq_len, num_domains, useTime, useLength, useDirection, useTcp, useQuic, useBurst)
df = DFNet().build((seq_len, 1), num_domains)
varcnn = varcnn(seq_len, num_domains)

early_stopping = EarlyStopping(monitor='val_accuracy', patience = 10, verbose=1)

lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
if useBurst:
    history= lstm.fit([X_train[:,:,:-1], X_train[:,:,-1]], y_train, validation_split=0.15, epochs=100, batch_size=32, use_multiprocessing=True, callbacks=[early_stopping], workers=20)
else:
    history= lstm.fit(X_train, y_train, validation_split=0.15, epochs=100, batch_size=32, use_multiprocessing=True, callbacks=[early_stopping], workers=20)
if useBurst:
    test_acc = lstm.evaluate(x=[X_test[:,:,:-1],X_test[:,:,-1]], y=y_test)
else:
    test_acc = model.evaluate(X_test, y=y_test)
print('lstm:', test_acc)

df.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history= df.fit(X_train[:,:,3], y_train, validation_split=0.15, epochs=100, batch_size=32, use_multiprocessing=True, callbacks=[early_stopping], workers=20)
test_acc = df.evaluate(X_test[:,:,3], y=y_test)
print('df:', test_acc)

varcnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history= varcnn.fit([X_train[:,:,2], X_train[:,:,3]], y_train, validation_split=0.15, epochs=100, batch_size=32, use_multiprocessing=True, callbacks=[early_stopping], workers=20)
test_acc = varcnn.evaluate([X_test[:,:,2], X_test[:,:,3]], y=y_test)
print('varcnn:', test_acc)

