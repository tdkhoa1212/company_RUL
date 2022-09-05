from tensorflow.keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
import tensorflow as  tf
from keras.models import Model

def autoencoder_model(type_, training=False):
    if type_ == 'PHM':
      inputs = Input(shape=(2, 2560))
      x1 = 2
      x2 = 2560
    else:
      inputs = Input(shape=(2, 32768))
      x1 = 2
      x2 = 32768
    L1 = LSTM(1024, activation='tanh', return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.00))(inputs)
    L1 = Dropout(0.2)(L1)
    L2 = LSTM(32, activation='tanh', return_sequences=False)(L1)
    L2 = Dropout(0.2)(L2)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(32, activation='tanh', return_sequences=True)(L3)
    L4 = Dropout(0.2)(L4)
    L5 = LSTM(1024, activation='tanh', return_sequences=True)(L4)
    L5 = Dropout(0.2)(L5)
    output = TimeDistributed(Dense(x2))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model
