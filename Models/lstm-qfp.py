from keras.layers import Dense, LSTM, Input, Concatenate, Dropout
from keras.models import Model

def lstm_qfp(useTime, useLength, useDirection, useTcp, useQuic, useBurst):
    num_features = sum([useTime,useLength,useDirection, useTcp, useQuic])
    
	input_data_1 = Input(name='5Features', shape=(newseqlen,num_features))
    
    five = LSTM(units=64,activation='tanh',recurrent_activation='sigmoid',return_sequences=True,kernel_regularizer='l1')(input_data_1)
    five = LSTM(units=64,activation='tanh',recurrent_activation='sigmoid',return_sequences=False)(five)
    
    if useBurst:
        input_data_2 = Input(name='Burst', shape=(newseqlen,1))
        burst = LSTM(units=64,activation='tanh',recurrent_activation='sigmoid',return_sequences=True,kernel_regularizer='l1')(input_data_2)
        burst = LSTM(units=64,activation='tanh',recurrent_activation='sigmoid',return_sequences=False)(burst)
    
    if useBurst:
        concatted = Concatenate()([five, burst])
    else:
        concatted = Concatenate()([five])
    x = Dropout(0.2, name='dropout_2')(concatted)
    out = Dense(units=y_train.shape[1], activation='softmax', name='softmax')(x)
    model = Model(inputs=[input_data_1,input_data_2], outputs=out)
    return model