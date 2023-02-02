from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from lstmqfp import lstm_qfp
from read_data import get_data
import json
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, auc
import pickle

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

lstm = keras.models.load_model('Models/allmodel2')
df = keras.models.load_model('Models/dfmodel2')
varcnn = keras.models.load_model('Models/varcnnmodel2')
models = [[lstm, 'all'], [df, 'df'], [varcnn, 'varcnn']]

for model, name in models:
    print(name)
    if name == 'all':
        y_pred = model.predict([X_test[:,:,:-1], X_test[:,:,-1]])
    
    if name == 'varcnn':
        y_pred = model.predict([X_test[:,:,1], X_test[:,:,2]])
    
    if name == 'df':
        y_pred = model.predict(X_test[:,:,2])
    
    y_score = y_pred
    
    y_onehot_test = y_test
    
    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.10f}")
    
    n_classes = num_domains
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    
    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)
    
    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
    
    # Average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.10f}")
    
    with open('Results/fprmodel2'+name+'.pkl', 'wb') as f:
        pickle.dump(fpr, f)
    
    with open('Results/tprmodel2'+name+'.pkl', 'wb') as f:
        pickle.dump(tpr, f)
    
    with open('Results/rocmodel2'+name+'.pkl', 'wb') as f:
        pickle.dump(roc_auc, f)
    

