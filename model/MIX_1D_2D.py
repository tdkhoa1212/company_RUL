from tensorflow.keras.layers import Conv1D, Activation, Dense, concatenate, BatchNormalization, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda, GlobalAveragePooling2D, ReLU, MaxPooling2D, Flatten, Dropout, LSTM
import tensorflow as tf
from keras.models import Model
from keras import layers, regularizers
import keras.backend as K

def fully_concatenate(hidden_out_1D, hidden_out_2D, hidden_out_extracted, training):
    all_ = concatenate((hidden_out_1D, hidden_out_2D, hidden_out_extracted))
    return all_

def mix_model(opt, cnn_1d_model, resnet_50, lstm_extracted_model, input_1D, input_2D, input_extracted, training=False):
  out_1D = cnn_1d_model(opt, training, input_1D)
  out_2D = resnet_50(opt)(input_2D, training=training)
  out_extracted = lstm_extracted_model(opt, training, input_extracted)
  
  network_1D = Model(input_1D, out_1D, name='network_1D')
  network_2D = Model(input_2D, out_2D, name='network_2D')
  network_extracted = Model(input_extracted, out_extracted, name='network_extracted')
  
  hidden_out_1D = network_1D([input_1D])
  hidden_out_2D = network_2D([input_2D])
  hidden_out_extracted = network_extracted([input_extracted])
  
  merged_value_1 = fully_concatenate(hidden_out_1D, hidden_out_2D, hidden_out_extracted, training)
    
  Condition = Dense(3, 
                    activation='softmax', 
                    name='Condition')(merged_value_1)
  RUL = Dense(1, 
              activation='sigmoid', 
              name='RUL')(merged_value_1)
  return Condition, RUL
